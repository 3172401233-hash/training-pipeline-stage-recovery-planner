# training-pipeline-stage-recovery-planner
规划训练流水线失败后的阶段重放和产物复用

## 概述

`recovery_planner.py` 根据阶段依赖、参数、代码版本、输入/输出摘要、失败点、
上传状态和版本指针，生成一份确定性的恢复计划：哪些阶段可以复用缓存产物，
哪些必须重放，哪些只需补传产物。

**本模块只做规划，不直接提交训练任务。** 它输出 `RecoveryPlan`，由外部执行器
按计划重放阶段、补传产物，并在全部产物就绪后原子切换版本指针。

## 复用判定

阶段缓存可复用（`reuse`）当且仅当：

- 上一次运行成功，且指纹一致（输入摘要、参数、代码版本、父阶段指纹全部相同）；
- 所有输出产物摘要非空、未过期（`expires_at` 晚于规划时刻）；
- 所有产物已完整上传（否则为 `reupload`，只补传不重跑）。

以下情况必须重放（`replay`）：从未运行、上次失败、指纹变化、产物摘要缺失、
缓存过期，或任一父阶段需要重放（重放沿依赖图向下游传播，分叉/汇聚结构同样
适用）。版本指针指向的阶段产物若与当前指纹不符（多版本历史产物），同样重放。

## 原子切换

计划的 `publishes` 是一组必须原子提交的指针切换：只有当所有 `replay` /
`reupload` 阶段完成且产物全部上传后，才把 `final_model`、`eval_results` 等
指针一次性切到新版本。切换前旧版本持续对外服务，重放失败不会产生半更新状态。

## 幂等性

阶段版本号由指纹派生（`v-<fingerprint前16位>`），同一快照的规划结果完全
确定。重复恢复收敛到同一计划；指针已指向目标版本时不再生成发布步骤。

## 用法

```python
from recovery_planner import (
    ArtifactState, PipelineSnapshot, StageRun, StageSpec,
    VersionPointer, plan_recovery,
)

snapshot = PipelineSnapshot(
    stages={
        "preprocess": StageSpec(name="preprocess", code_version="prep@1",
                                inputs={"raw": "raw-001"}),
        "train": StageSpec(name="train", depends_on=("preprocess",),
                           params={"lr": 0.01}, code_version="train@3",
                           inputs={"dataset": "data-v1"}),
    },
    runs={
        "preprocess": StageRun(stage="preprocess", status="succeeded",
                               fingerprint="<上次运行指纹>",
                               outputs={"out": ArtifactState(digest="d1")}),
        "train": StageRun(stage="train", status="failed"),
    },
    pointers={"final_model": VersionPointer("final_model", "v-old")},
    pointer_stage={"final_model": "train"},
)
plan = plan_recovery(snapshot, now=time.time())
print(plan.stages_to_run())  # 需要重放/补传的阶段
print(plan.publishes)        # 待原子提交的指针切换
```

## 测试

```
python -m unittest
```

覆盖失败重跑、摘要损坏、缓存过期、部分上传、分叉依赖、多版本产物与重复恢复。
兼容 Python 3.8.2+。
