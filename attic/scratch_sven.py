from src.m_fac_tf.utils.datasets import load_model, load_dataset, load_optimizer


import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf


#@hydra.main(version_base="1.1", config_path="conf", config_name="mfac_experiment")  # noqa E501
#def optimizer_experiment(cfg: DictConfig) -> None:
#    """Benchmark experiment"""
#    # print(hydra.utils.get_original_cwd())

if __name__ == "__main__":
    cfg = OmegaConf.load("conf/experiment1.yaml")
    experiment = cfg.experiments[0]
    experiment_name = experiment.get("name")
    experiment_optimizers = experiment.optimizer
    experiment_dataset = experiment.dataset
    train_data, test_data = load_dataset(experiment_dataset)
    experiment_model = experiment.model
    input_shape = train_data.element_spec[0]
    print(input_shape)
    #experiment_model = load_model(experiment_model, input_shape=)
    experiment_epochs = experiment.epochs
    # one experiment has several optimizers
    optimizer_names = list(experiment_optimizers.keys())
    optimizer_name = optimizer_names[0]
    optimizer_params = experiment_optimizers.get(optimizer_name).get("params")
    # depending on optimizer, batch size is set
    batch_size = experiment_optimizers.get(optimizer_name).get("name")
    optimizer = load_optimizer(name=optimizer_name,
                               params=optimizer_params)
    # fit model
    #experiment_model.compile(optimizer=optimizer,
    #                         loss="categorical_crossentropy",
    #                         metrics=["accuracy"])
    #experiment_model.fit(x=train_data,
    #                     batch_size=batch_size,
    #                    epochs=experiment_epochs)
    m = 5
    d = 8
    vec1 = tf.constant(1, shape=(1, d), dtype=tf.float32)
    vec2 = tf.constant(2, shape=(1, d), dtype=tf.float32)
    vec3 = tf.constant(3, shape=(1, d), dtype=tf.float32)
    vec4 = tf.constant(4, shape=(1, d), dtype=tf.float32)
    vec5 = tf.constant(5, shape=(1, d), dtype=tf.float32)
    vec6 = tf.constant(6, shape=(1, d), dtype=tf.float32)
    vec7 = tf.constant(7, shape=(1, d), dtype=tf.float32)

    mat = RowWiseMatrixFifo(m=m)
    mat.append(vec1)
    mat.append(vec2)
    mat.append(vec3)
    mat.append(vec4)
    mat.append(vec5)
    print(mat.values)
    mat.append(vec6)
    print(mat.values)
    print(mat.counter)
    mat.counter = 0
    mat.values = None
    mat.append(vec7)
    print(mat.values)
    #print(vec1.shape[1])
    #values = tf.Variable(tf.zeros(shape=[m, vec1.shape[1]]))
    #values[m - 1, :].assign(vec6)
    #values[0, :].assign(vec1)
    #print(values)
    #maintained_values = tf.identity(values[:m - 1, :])
    #print(maintained_values)
    #values[1:, :].assign(maintained_values)
    #values[0, :].assign(vec4)
    #print(values)
    #gradients_list = [tf.Variable(tf.random.normal((5, 2, 19))), tf.Variable(tf.random.normal((20,)))]
    #flatten_grads = list(map(lambda x: tf.reshape(x, [-1]), gradients_list))
    #flatten_grads = tf.concat(flatten_grads, axis = 0).numpy()
    #shapes_grads = list(map(lambda x: x.shape, gradients_list))
    #deflattened = deflatten(flatten_grads, shapes_grads)
    #print([var.shape for var in deflattened])


    
