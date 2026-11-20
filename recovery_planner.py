"""Training pipeline stage recovery planner.

Given stage dependencies, parameters, code versions, input/output digests,
failure points, upload status and version pointers, decide which stages may
reuse their cached artifacts and which must be replayed.

A stage's cache is reusable only when its inputs, parameters, code version
and all parent artifacts are unchanged. Final model and evaluation result
pointers are switched atomically in a single commit step.

This module only *plans* recovery. It never submits training jobs itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

ACTION_REUSE = "reuse"
ACTION_REPLAY = "replay"
ACTION_REUPLOAD = "reupload"

STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_NEVER_RUN = "never_run"


class PlanError(ValueError):
    """Raised when the pipeline snapshot is inconsistent."""


@dataclass(frozen=True)
class StageSpec:
    """Static definition of a pipeline stage."""

    name: str
    depends_on: Tuple[str, ...] = ()
    params: Mapping[str, object] = field(default_factory=dict)
    code_version: str = ""
    inputs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactState:
    """Recorded state of one output artifact from a previous run."""

    digest: str
    uploaded: bool = True
    expires_at: Optional[float] = None

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and self.expires_at <= now


@dataclass(frozen=True)
class StageRun:
    """Recorded outcome of a stage's previous run."""

    stage: str
    status: str = STATUS_NEVER_RUN
    fingerprint: Optional[str] = None
    outputs: Mapping[str, ArtifactState] = field(default_factory=dict)


@dataclass(frozen=True)
class VersionPointer:
    """Atomic pointer to the currently published artifact version."""

    name: str
    current_version: Optional[str] = None


@dataclass(frozen=True)
class PipelineSnapshot:
    """Everything the planner needs to know about the pipeline."""

    stages: Mapping[str, StageSpec]
    runs: Mapping[str, StageRun] = field(default_factory=dict)
    pointers: Mapping[str, VersionPointer] = field(default_factory=dict)
    # pointer name -> stage whose output feeds that pointer
    pointer_stage: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StageDecision:
    stage: str
    action: str
    reason: str
    fingerprint: str
    version: str


@dataclass(frozen=True)
class PublishStep:
    """One pointer switch inside the atomic commit."""

    pointer: str
    from_version: Optional[str]
    to_version: str


@dataclass(frozen=True)
class RecoveryPlan:
    """Ordered stage decisions plus a single atomic publish commit.

    The publishes must be applied atomically, and only after every
    replay/reupload decision has completed and its artifacts are fully
    uploaded. Until the commit happens, existing pointers keep serving
    the old versions.
    """

    decisions: Tuple[StageDecision, ...]
    publishes: Tuple[PublishStep, ...]

    def decision_for(self, stage: str) -> StageDecision:
        for decision in self.decisions:
            if decision.stage == stage:
                return decision
        raise KeyError(stage)

    def stages_to_run(self) -> Tuple[str, ...]:
        return tuple(
            d.stage for d in self.decisions if d.action != ACTION_REUSE
        )


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(spec: StageSpec, parent_fingerprints: Mapping[str, str]) -> str:
    payload = {
        "stage": spec.name,
        "params": spec.params,
        "code_version": spec.code_version,
        "inputs": dict(spec.inputs),
        "parents": {name: parent_fingerprints[name] for name in spec.depends_on},
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _version_for(fingerprint: str) -> str:
    return "v-" + fingerprint[:16]


def _topological_order(stages: Mapping[str, StageSpec]) -> Tuple[str, ...]:
    order = []
    permanent = set()
    temporary = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise PlanError("dependency cycle involving stage %r" % name)
        if name not in stages:
            raise PlanError("unknown stage %r" % name)
        temporary.add(name)
        for parent in stages[name].depends_on:
            if parent not in stages:
                raise PlanError(
                    "stage %r depends on unknown stage %r" % (name, parent)
                )
            visit(parent)
        temporary.discard(name)
        permanent.add(name)
        order.append(name)

    for name in stages:
        visit(name)
    return tuple(order)


def _decide_stage(
    run: Optional[StageRun],
    fingerprint: str,
    now: float,
    parents_replaying: bool,
) -> Tuple[str, str]:
    """Return (action, reason) for a single stage."""
    if parents_replaying:
        return ACTION_REPLAY, "parent stage must replay"
    if run is None or run.status == STATUS_NEVER_RUN:
        return ACTION_REPLAY, "never ran"
    if run.status != STATUS_SUCCEEDED:
        return ACTION_REPLAY, "previous run %s" % run.status
    if run.fingerprint != fingerprint:
        return ACTION_REPLAY, "inputs, params, code version or parent artifacts changed"

    artifacts = list(run.outputs.values())
    if not artifacts:
        return ACTION_REPLAY, "no recorded output artifacts"
    for artifact in artifacts:
        if not artifact.digest:
            return ACTION_REPLAY, "output artifact has missing/empty digest"
    for artifact in artifacts:
        if artifact.is_expired(now):
            return ACTION_REPLAY, "cached artifact expired"
    missing_uploads = [a for a in artifacts if not a.uploaded]
    if missing_uploads:
        return ACTION_REUPLOAD, "%d artifact(s) not fully uploaded" % len(missing_uploads)
    return ACTION_REUSE, "cache valid"


def plan_recovery(snapshot: PipelineSnapshot, now: float = 0.0) -> RecoveryPlan:
    """Build a deterministic recovery plan for the given snapshot.

    Planning is idempotent: the same snapshot always yields the same
    decisions, versions and publish steps, so duplicate recovery attempts
    converge instead of duplicating work.
    """
    order = _topological_order(snapshot.stages)

    decisions = []
    fingerprints: Dict[str, str] = {}
    actions: Dict[str, str] = {}
    for name in order:
        spec = snapshot.stages[name]
        parent_fps = {parent: fingerprints[parent] for parent in spec.depends_on}
        fingerprint = _fingerprint(spec, parent_fps)
        fingerprints[name] = fingerprint

        run = snapshot.runs.get(name)
        if run is not None and run.stage != name:
            raise PlanError("run record %r stored under stage %r" % (run.stage, name))
        parents_replaying = any(
            actions[parent] == ACTION_REPLAY for parent in spec.depends_on
        )
        action, reason = _decide_stage(run, fingerprint, now, parents_replaying)
        actions[name] = action
        decisions.append(
            StageDecision(
                stage=name,
                action=action,
                reason=reason,
                fingerprint=fingerprint,
                version=_version_for(fingerprint),
            )
        )

    publishes = []
    for pointer_name in sorted(snapshot.pointer_stage):
        stage = snapshot.pointer_stage[pointer_name]
        if stage not in fingerprints:
            raise PlanError(
                "pointer %r references unknown stage %r" % (pointer_name, stage)
            )
        target_version = _version_for(fingerprints[stage])
        pointer = snapshot.pointers.get(pointer_name)
        current = pointer.current_version if pointer is not None else None
        if current != target_version:
            publishes.append(
                PublishStep(
                    pointer=pointer_name,
                    from_version=current,
                    to_version=target_version,
                )
            )

    return RecoveryPlan(decisions=tuple(decisions), publishes=tuple(publishes))
