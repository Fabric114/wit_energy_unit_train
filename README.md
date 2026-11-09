```text
.
├── README.md
├── configs                        //配置文件
│   ├── capture.yaml               //采集
│   └── train
│       ├── yolo11n.yaml           //单阶段训练配置
│       └── yolo11n_two_stage.yaml //两阶段训练配置
├── energy_train                   //核心代码
│   ├── annotation_cli.py          //build / render 共用的命令行逻辑
│   ├── annotations.py             //标注校验数据集
│   ├── capture.py                 //采集图像（目前仅d435）
│   ├── common.py                  //通用
│   ├── dataset.py                 //数据集检查、划分、构建
│   ├── group_proposal.py          //按时间断点生成分组草稿
│   ├── snapshot.py                //已发布快照校验（训练共用）
│   ├── splitting.py               //场景份、group-file、份分配
│   └── training.py                //训练配置校验、provenance 记录
├── environment.yml                //conda 环境 wit_energy_unit
├── raw                            //采集的数据
│   └── ...
├── reference_model                //参考模型 .pt 与 OpenVINO 模型
│   └── ...
├── requirements
│   ├── capture.txt                //采集与标注依赖
│   └── train.txt                  //训练依赖
├── script                         //命令行入口
│   ├── build_dataset.py
│   ├── capture_d435.py
│   ├── doctor.py
│   ├── propose_groups.py
│   ├── prelabel_openvino.py
│   ├── render_annotations.py
│   └── train.py
└── tests                          //unittest 测试
```

TODO：`datasets/`（数据集）、`runs/`（训练输出）、`reports/`（检查图）、`weights/pretrained/`（训练起点权重）

## 命令

在项目根目录执行，先激活环境：`conda activate wit_energy_unit`。

```bash
# 环境检查
python script/doctor.py
python script/doctor.py --camera                 # 枚举相机

# 采集；同一段连续拍摄使用同一个 group-id
python script/capture_d435.py --session s03_near_front --group-id g03

# 用 OpenVINO 模型生成预标注 JSON
python script/prelabel_openvino.py --session s03

# 校验标注并检查划分可行性
python script/build_dataset.py --check-only

# 可选：按拍摄间隔生成分组草稿（只读 raw，confirmed: false，需人工看图核对）
python script/propose_groups.py --output configs/dataset/groups.yaml
# 核对后把 groups.yaml 中的 confirmed 改为 true，再检查和构建
python script/build_dataset.py --group-file configs/dataset/groups.yaml --check-only
python script/build_dataset.py --group-file configs/dataset/groups.yaml --output datasets/energy_v2

# 构建 YOLO 数据集，输出必须是新目录
python script/build_dataset.py --output datasets/energy_v1

# 可选：导出检测框检查图到新目录
python script/render_annotations.py --session s01 --output reports/s01_check

# 训练：先只校验，确认后再正式训练
python script/train.py --config configs/train/yolo11n.yaml --data datasets/energy_v1/data.yaml --check-only
python script/train.py --config configs/train/yolo11n.yaml --data datasets/energy_v1/data.yaml

# 运行测试
python -m unittest discover -s tests
```
