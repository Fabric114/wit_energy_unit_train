```text
.
├── README.md
├── configs                  //配置文件
│   └── ...
├── requirements             //依赖
│   └── ...
├── raw                      //采集图片保存目录
│   └── ...
├── energy_train
│   ├── annotation_cli.py
│   ├── annotations.py      //校验json
│   ├── capture.py          //采集图像，目前仅d435
│   ├── common.py           //通用
│   └── dataset.py          //构建数据集
├── environment.yml         //conda环境配置
├── script                  //实际运行入口
│   ├── build_dataset.py    //构建数据集 --output datasets/xxx
│   ├── capture_d435.py     //d435采集脚本
│   └── doctor.py           //环境检查脚本
└── tests
    ├── test_annotations.py
    ├── test_capture.py
    └── test_dataset.py
```
