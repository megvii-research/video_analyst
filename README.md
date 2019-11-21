
```
├── experiments  # 存放大家实验的地方，yaml格式
├── video_analyst
│   ├── data # 数据相关模块
│   │   ├── dataset # 数据集解析管理
│   │   ├── sampler # 数据采样 包含数据内部采样和数据集之间的采样
│   │   ├── dataloader.py # 负责供数据的逻辑
│   │   └── transformer # 数据增强 
│   ├── engine # 流程控制, 负责整个训练的进行，如何加载参数/模型
│   │   ├── hook # 一些用于训练过程中的处理的hook，训练中间的可视化，log，测试过程中log，可视化和指标
│   │   ├── trainer.py # 完成一个epoch的训练
│   │   ├── tester.py # 完成一个测试
│   ├── model # 模型搭建模块
│   │   ├── backbone # 各种backbone
│   │   ├── task_model # 整体网络结构
│   │   ├── task_head # 业务输出结构
│   │   └── loss # 各种loss
│   ├── pipeline # 包含各种任务的pipeline，如track， vos等
│   ├── config # config 管理模块
│   ├── evaluation #  指标评价模块
│   |── optimize # 优化模块，lr，grad等
│   │   ├── lr_schedule # 学习率变化函数
│   │   ├── optimizer # 梯度优化方法, sgd等
│   │   ├── grad_modifier # freeze一些层呀之类的功能
├── README.md
├── main
│   ├── train.py # 模型训练入口 
│   ├── test.py # 模型测试入口 
└── utils # 实用工具
```