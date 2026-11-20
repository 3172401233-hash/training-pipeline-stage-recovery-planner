import unittest

from recovery_planner import (
    ACTION_REPLAY,
    ACTION_REUPLOAD,
    ACTION_REUSE,
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    ArtifactState,
    PipelineSnapshot,
    PlanError,
    StageRun,
    StageSpec,
    VersionPointer,
    plan_recovery,
)

NOW = 1_000_000.0


def linear_stages(train_input_digest="data-v1"):
    return {
        "preprocess": StageSpec(
            name="preprocess",
            params={"normalize": True},
            code_version="prep@1",
            inputs={"raw": "raw-001"},
        ),
        "train": StageSpec(
            name="train",
            depends_on=("preprocess",),
            params={"lr": 0.01, "epochs": 10},
            code_version="train@3",
            inputs={"dataset": train_input_digest},
        ),
        "evaluate": StageSpec(
            name="evaluate",
            depends_on=("train",),
            params={"split": "test"},
            code_version="eval@2",
        ),
    }


def linear_snapshot(runs, pointers=None, train_input_digest="data-v1"):
    return PipelineSnapshot(
        stages=linear_stages(train_input_digest),
        runs=runs,
        pointers=pointers
        or {
            "final_model": VersionPointer("final_model"),
            "eval_results": VersionPointer("eval_results"),
        },
        pointer_stage={"final_model": "train", "eval_results": "evaluate"},
    )


def fingerprints_for(snapshot):
    """Learn the planned fingerprints by planning against an empty history."""
    empty = PipelineSnapshot(
        stages=snapshot.stages,
        runs={},
        pointers=snapshot.pointers,
        pointer_stage=snapshot.pointer_stage,
    )
    plan = plan_recovery(empty, now=NOW)
    return {d.stage: d.fingerprint for d in plan.decisions}


def succeeded_run(stage, fingerprint, digest=None, uploaded=True, expires_at=None):
    return StageRun(
        stage=stage,
        status=STATUS_SUCCEEDED,
        fingerprint=fingerprint,
        outputs={
            "model.bin": ArtifactState(
                digest=digest if digest is not None else "out-" + stage,
                uploaded=uploaded,
                expires_at=expires_at,
            )
        },
    )


def make_runs(snapshot, succeeded, failed=(), **artifact_overrides):
    fps = fingerprints_for(snapshot)
    runs = {}
    for stage in succeeded:
        overrides = artifact_overrides.get(stage, {})
        runs[stage] = succeeded_run(stage, fps[stage], **overrides)
    for stage in failed:
        runs[stage] = StageRun(stage=stage, status=STATUS_FAILED, fingerprint=fps[stage])
    return runs


class FailureRerunTest(unittest.TestCase):
    def test_failed_train_replays_only_train_and_eval(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(snapshot, succeeded=("preprocess",), failed=("train",))
        plan = plan_recovery(
            PipelineSnapshot(
                stages=snapshot.stages,
                runs=runs,
                pointers=snapshot.pointers,
                pointer_stage=snapshot.pointer_stage,
            ),
            now=NOW,
        )
        self.assertEqual(plan.decision_for("preprocess").action, ACTION_REUSE)
        self.assertEqual(plan.decision_for("train").action, ACTION_REPLAY)
        self.assertEqual(plan.decision_for("evaluate").action, ACTION_REPLAY)
        self.assertEqual(plan.stages_to_run(), ("train", "evaluate"))

    def test_full_reuse_when_everything_succeeded(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(snapshot, succeeded=("preprocess", "train", "evaluate"))
        snapshot = linear_snapshot(runs=runs)
        plan = plan_recovery(snapshot, now=NOW)
        self.assertEqual(plan.stages_to_run(), ())
        # Pointers were never published, so the plan publishes the reused
        # versions; nothing is replayed to get there.
        by_pointer = {p.pointer: p for p in plan.publishes}
        self.assertIsNone(by_pointer["final_model"].from_version)
        self.assertEqual(
            by_pointer["final_model"].to_version,
            plan.decision_for("train").version,
        )


class DigestAndParamChangeTest(unittest.TestCase):
    def test_changed_input_digest_invalidates_stage_and_children(self):
        old_snapshot = linear_snapshot(runs={}, train_input_digest="data-v1")
        runs = make_runs(old_snapshot, succeeded=("preprocess", "train", "evaluate"))
        # Input digest moved on: eval must not reuse stale artifacts.
        new_snapshot = linear_snapshot(runs=runs, train_input_digest="data-v2")
        plan = plan_recovery(new_snapshot, now=NOW)
        self.assertEqual(plan.decision_for("preprocess").action, ACTION_REUSE)
        self.assertEqual(plan.decision_for("train").action, ACTION_REPLAY)
        self.assertEqual(plan.decision_for("evaluate").action, ACTION_REPLAY)

    def test_param_change_replays_only_affected_branch(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(snapshot, succeeded=("preprocess", "train", "evaluate"))
        stages = linear_stages()
        stages["train"] = StageSpec(
            name="train",
            depends_on=("preprocess",),
            params={"lr": 0.1, "epochs": 10},
            code_version="train@3",
            inputs={"dataset": "data-v1"},
        )
        changed = PipelineSnapshot(
            stages=stages,
            runs=runs,
            pointers=snapshot.pointers,
            pointer_stage=snapshot.pointer_stage,
        )
        plan = plan_recovery(changed, now=NOW)
        self.assertEqual(plan.decision_for("preprocess").action, ACTION_REUSE)
        self.assertEqual(plan.decision_for("train").action, ACTION_REPLAY)
        self.assertEqual(plan.decision_for("evaluate").action, ACTION_REPLAY)

    def test_bad_output_digest_forces_replay(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(
            snapshot,
            succeeded=("preprocess", "train", "evaluate"),
            train={"digest": ""},
        )
        snapshot = linear_snapshot(runs=runs)
        plan = plan_recovery(snapshot, now=NOW)
        self.assertEqual(plan.decision_for("preprocess").action, ACTION_REUSE)
        self.assertEqual(plan.decision_for("train").action, ACTION_REPLAY)
        self.assertEqual(plan.decision_for("evaluate").action, ACTION_REPLAY)
        self.assertIn("digest", plan.decision_for("train").reason)


class CacheExpiryAndUploadTest(unittest.TestCase):
    def test_expired_cache_replays(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(
            snapshot,
            succeeded=("preprocess", "train", "evaluate"),
            preprocess={"expires_at": NOW - 1},
        )
        snapshot = linear_snapshot(runs=runs)
        plan = plan_recovery(snapshot, now=NOW)
        self.assertEqual(plan.decision_for("preprocess").action, ACTION_REPLAY)
        self.assertEqual(plan.decision_for("train").action, ACTION_REPLAY)
        self.assertEqual(plan.decision_for("evaluate").action, ACTION_REPLAY)

    def test_unexpired_cache_reuses(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(
            snapshot,
            succeeded=("preprocess", "train", "evaluate"),
            preprocess={"expires_at": NOW + 3600},
        )
        snapshot = linear_snapshot(runs=runs)
        plan = plan_recovery(snapshot, now=NOW)
        self.assertEqual(plan.stages_to_run(), ())

    def test_partial_upload_reuploads_without_replay(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(
            snapshot,
            succeeded=("preprocess", "train", "evaluate"),
            train={"uploaded": False},
        )
        snapshot = linear_snapshot(runs=runs)
        plan = plan_recovery(snapshot, now=NOW)
        self.assertEqual(plan.decision_for("preprocess").action, ACTION_REUSE)
        self.assertEqual(plan.decision_for("train").action, ACTION_REUPLOAD)
        # Downstream artifacts are still valid; they only need the parent
        # upload to complete before the atomic publish.
        self.assertEqual(plan.decision_for("evaluate").action, ACTION_REUSE)


class ForkedDependencyTest(unittest.TestCase):
    def diamond_snapshot(self, runs, branch_b_lr=0.01):
        stages = {
            "preprocess": StageSpec(
                name="preprocess", code_version="prep@1", inputs={"raw": "raw-001"}
            ),
            "branch_a": StageSpec(
                name="branch_a",
                depends_on=("preprocess",),
                params={"lr": 0.01},
                code_version="train@3",
            ),
            "branch_b": StageSpec(
                name="branch_b",
                depends_on=("preprocess",),
                params={"lr": branch_b_lr},
                code_version="train@3",
            ),
            "select_best": StageSpec(
                name="select_best",
                depends_on=("branch_a", "branch_b"),
                code_version="select@1",
            ),
        }
        return PipelineSnapshot(
            stages=stages,
            runs=runs,
            pointers={"final_model": VersionPointer("final_model")},
            pointer_stage={"final_model": "select_best"},
        )

    def test_diamond_replays_only_changed_branch_and_join(self):
        snapshot = self.diamond_snapshot(runs={})
        runs = make_runs(
            snapshot, succeeded=("preprocess", "branch_a", "branch_b", "select_best")
        )
        changed = self.diamond_snapshot(runs=runs, branch_b_lr=0.5)
        plan = plan_recovery(changed, now=NOW)
        self.assertEqual(plan.decision_for("preprocess").action, ACTION_REUSE)
        self.assertEqual(plan.decision_for("branch_a").action, ACTION_REUSE)
        self.assertEqual(plan.decision_for("branch_b").action, ACTION_REPLAY)
        self.assertEqual(plan.decision_for("select_best").action, ACTION_REPLAY)

    def test_cycle_is_rejected(self):
        stages = {
            "a": StageSpec(name="a", depends_on=("b",)),
            "b": StageSpec(name="b", depends_on=("a",)),
        }
        with self.assertRaises(PlanError):
            plan_recovery(PipelineSnapshot(stages=stages), now=NOW)

    def test_unknown_dependency_is_rejected(self):
        stages = {"a": StageSpec(name="a", depends_on=("ghost",))}
        with self.assertRaises(PlanError):
            plan_recovery(PipelineSnapshot(stages=stages), now=NOW)


class AtomicPublishTest(unittest.TestCase):
    def test_publish_switches_both_pointers_to_new_versions(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(snapshot, succeeded=("preprocess",), failed=("train",))
        snapshot = linear_snapshot(
            runs=runs,
            pointers={
                "final_model": VersionPointer("final_model", "v-oldmodel"),
                "eval_results": VersionPointer("eval_results", "v-oldeval"),
            },
        )
        plan = plan_recovery(snapshot, now=NOW)
        self.assertEqual(len(plan.publishes), 2)
        by_pointer = {p.pointer: p for p in plan.publishes}
        self.assertEqual(by_pointer["final_model"].from_version, "v-oldmodel")
        self.assertEqual(
            by_pointer["final_model"].to_version,
            plan.decision_for("train").version,
        )
        self.assertEqual(
            by_pointer["eval_results"].to_version,
            plan.decision_for("evaluate").version,
        )
        # Old versions are never deleted by the plan; the pointer switch is
        # the only visible change, so a failed replay leaves v-old* serving.
        self.assertNotEqual(
            by_pointer["final_model"].to_version, "v-oldmodel"
        )

    def test_pointer_already_at_target_is_not_republished(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(snapshot, succeeded=("preprocess", "train", "evaluate"))
        fps = fingerprints_for(linear_snapshot(runs={}))
        snapshot = linear_snapshot(
            runs=runs,
            pointers={
                "final_model": VersionPointer(
                    "final_model", "v-" + fps["train"][:16]
                ),
                "eval_results": VersionPointer(
                    "eval_results", "v-" + fps["evaluate"][:16]
                ),
            },
        )
        plan = plan_recovery(snapshot, now=NOW)
        self.assertEqual(plan.publishes, ())

    def test_multiversion_history_picks_current_fingerprint_version(self):
        # Several historical runs of train exist; only the record matching
        # the current fingerprint may be reused.
        snapshot = linear_snapshot(runs={})
        fps = fingerprints_for(snapshot)
        stale_run = StageRun(
            stage="train",
            status=STATUS_SUCCEEDED,
            fingerprint="stale-fingerprint-from-older-code",
            outputs={"model.bin": ArtifactState(digest="old-digest")},
        )
        runs = make_runs(snapshot, succeeded=("preprocess", "evaluate"))
        runs["train"] = stale_run
        snapshot = linear_snapshot(runs=runs)
        plan = plan_recovery(snapshot, now=NOW)
        decision = plan.decision_for("train")
        self.assertEqual(decision.action, ACTION_REPLAY)
        self.assertEqual(decision.version, "v-" + fps["train"][:16])


class DuplicateRecoveryTest(unittest.TestCase):
    def test_planning_is_deterministic_and_idempotent(self):
        snapshot = linear_snapshot(runs={})
        runs = make_runs(snapshot, succeeded=("preprocess",), failed=("train",))
        snapshot = linear_snapshot(runs=runs)
        first = plan_recovery(snapshot, now=NOW)
        second = plan_recovery(snapshot, now=NOW)
        self.assertEqual(first, second)

    def test_recovery_after_recovery_converges(self):
        # Simulate a duplicate recovery attempt after the first one finished:
        # the previously failed stage now has a successful record with the
        # planned fingerprint, so nothing reruns and nothing republishes.
        snapshot = linear_snapshot(runs={})
        fps = fingerprints_for(snapshot)
        runs = make_runs(snapshot, succeeded=("preprocess", "train", "evaluate"))
        snapshot = linear_snapshot(
            runs=runs,
            pointers={
                "final_model": VersionPointer(
                    "final_model", "v-" + fps["train"][:16]
                ),
                "eval_results": VersionPointer(
                    "eval_results", "v-" + fps["evaluate"][:16]
                ),
            },
        )
        plan = plan_recovery(snapshot, now=NOW)
        self.assertEqual(plan.stages_to_run(), ())
        self.assertEqual(plan.publishes, ())


if __name__ == "__main__":
    unittest.main()
