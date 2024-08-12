# Deep Learning-Based Super-Resolution and Visualization of Ocean Internal Waves

采用多种设计模式构建项目，包括工厂方法、单例模式、建造者模式、适配器模式、桥接模式、组合模式等，形成易拓展，精炼，健壮的代码工程。

- 海洋内波超分项目链接：https://github.com/spcrey/attention-isw-hr
- 海洋数据可视化工具包链接：https://github.com/spcrey/odvpkg
- 内波可视化项目链接：https://github.com/spcrey/isw-visual

```python
# 超分部分训练过程代码风格展示

param_receiver = TrainParamReceiver()
param = param_receiver()
# cuda
CudaManager(param.use_cuda, param.cuda_devices, param.batch_size_per_cuda)
# random
RandomManager(param.seed)
# recorder(include logger and metric)
logger = Logger("train", param.log_folder)
metric = TrainMetric(param.phy_fea_names, param.log_folder)
recorder = TrainRecorder(logger, metric)
recorder.logger.info("train start!")
recorder.program_backup()
recorder.param_backup(param)
# eage sampling and numerical preprocess of topography region (nptr)
croper = Croper(param.data_shape, param.sampling_crop_shape)
eage_point_generator_factory = EagePointGeneratorFactory()
eage_point_generator = eage_point_generator_factory(param.use_eage_sampling, param.eage_sampling_name, param.eage_sampling_random_scale, param.eage_sampling_eage_alpha)
grid_point_generator = GridPointGenerator()
train_sampler = Sampler(param.sampling_crop_shape, param.downsampling_rate, param.n_sampling_point, eage_point_generator)
eval_sampler = Sampler(param.sampling_crop_shape, param.downsampling_rate, param.n_sampling_point, grid_point_generator)
nptr = NptrFactory()(param.use_nptr, param.nptr_name)
# dataset(train and eval)
train_dataset = TrainDataset(param.train_dataset_folder, param.train_dataset_names, param.phy_fea_names, nptr, train_sampler, croper)
train_dataloader = TrainDataLoaderFactory()(train_dataset, param.n_sampling_crop)
eval_dataset = CropEvalDataset(param.train_dataset_folder, param.train_dataset_names, param.phy_fea_names, nptr, eval_sampler, croper)
eval_dataloader = CropEvalDataLoaderFactory()(eval_dataset, param.n_eval_sampling_crop)
# model, loss_fun and scheduler(optimizer)
factory = GridModelFactory()
grid_model = factory(param.grid_model_name, len(param.phy_fea_names), param.n_latent_fea, train_sampler.lres_crop_shape, param.n_grid_model_baselayer_fea, param.n_grid_model_layer)
factory = PointModelFactory()
point_model = factory(param.point_model_name, param.n_latent_fea, len(param.phy_fea_names), param.n_point_model_baselayer_fea, param.n_grid_model_layer)
pde_model = PdeModel(param.phy_fea_names, param.equation_file_path, train_dataset)
dual_model = TripleModel(grid_model, point_model, pde_model)
loss_fun = LossFunFactory()(param.loss_fun_type)
optimizer = OptimizerFactory()(param.optimizer_type, dual_model, param.learning_rate)
scheduler = Scheduler(optimizer)
# train, load and run
trainer = Trainer(train_dataloader, dual_model, loss_fun, optimizer, param.pde_loss_alpha)
evaler = CropEvaler(eval_dataloader, dual_model, param.eval_point_batch_size)
epoch_trainer = EpochTrainer(trainer, evaler, param.epoch_num, scheduler)
if param.resume_folder:
    epoch_trainer.load(param.resume_folder, param.resume_model_name)
epoch_trainer.run()

```

```bash
# 超分部分项目结构

├── README.md
├── approach
│   ├── cube_attention.py
│   ├── eage_point_generator_v1.py
│   └── fft_attention.py
├── equation
│   ├── convection.json
│   ├── isw.json
│   └── navier_stokes.json
├── eval.py
├── eval_param.json
├── src
│   ├── croper.py
│   ├── cuda_manager.py
│   ├── dataloader.py
│   ├── eage_point_generator_factory.py
│   ├── epoch_trainer.py
│   ├── evaler.py
│   ├── grid_data_process.py
│   ├── grid_model.py
│   ├── grid_model_factory.py
│   ├── logger.py
│   ├── loss_fun.py
│   ├── metric.py
│   ├── normalizer.py
│   ├── nptr.py
│   ├── ocean_dataset.py
│   ├── ocean_grid_data.py
│   ├── ocean_point_array.py
│   ├── optimizer.py
│   ├── param_receiver.py
│   ├── pde_model.py
│   ├── point_coord_iterator.py
│   ├── point_generator.py
│   ├── point_model.py
│   ├── point_model_factory.py
│   ├── random_manager.py
│   ├── recorder.py
│   ├── sampler.py
│   ├── singleton.py
│   ├── trainer.py
│   ├── triple_model.py
│   └── visualer.py
├── train.py
└── train_param.json
```
