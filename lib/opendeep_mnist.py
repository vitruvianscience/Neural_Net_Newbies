'''
OpenDeep MNIST Example / Tutorial

'''
from __future__ import print_function

from opendeep.log.logger import config_root_logger
from opendeep.models.container import Prototype
from opendeep.models.single_layer.basic import BasicLayer, SoftmaxLayer
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.data.standard_datasets.image.mnist import MNIST, datasets
from opendeep.monitor.monitor import Monitor
from opendeep.monitor.plot import Plot

# set up the logger to print everything to stdout and log files in opendeep/log/logs/
config_root_logger()

def create_mlp():
    # This method is to demonstrate adding layers one-by-one to a Prototype container.
    # As you can see, inputs_hook are created automatically by Prototype so we don't need to specify!
    mlp = Prototype()
    mlp.add(BasicLayer(input_size=28*28, output_size=512, activation='rectifier', noise='dropout'))
    mlp.add(BasicLayer(output_size=512, activation='rectifier', noise='dropout'))
    mlp.add(SoftmaxLayer(output_size=10))

    return mlp

def main():
    # grab our dataset (and don't concatenate together train and valid sets)
    mnist_dataset = MNIST(concat_train_valid=False)
    # create the mlp model from a Prototype
    mlp = create_mlp()
    # create an optimizer to train the model (stochastic gradient descent)
    optimizer = SGD(model=mlp,
                    dataset=mnist_dataset,
                    n_epoch=500,
                    batch_size=600,
                    learning_rate=.01,
                    momentum=.9,
                    nesterov_momentum=True,
                    save_frequency=500,
                    early_stop_threshold=0.997)
    # create a Monitor to view progress on a metric other than training cost
    error = Monitor('error', mlp.get_monitors()['softmax_error'], train=True, valid=True, test=True)
    # optionally, if you want to use Bokeh to view graphs of the cost and error make sure you have Bokeh
    # installed via pip. Then, start a bokeh server through the commmand: `bokeh-server`
    plot = None
    # uncomment this next line if you want to use Bokeh to graph your cost/error
    # plot = Plot("OpenDeep MLP Example", monitor_channels=error, open_browser=True)

    # train it! feel free to do a KeyboardInterrupt - it will save the latest parameters.
    optimizer.train(monitor_channels=error, plot=plot)

    # to evaluate our newly-trained model, us the .run() function with some data input!
    n_examples = 50
    test_data, test_labels = mnist_dataset.getSubset(datasets.TEST)
    predictions = mlp.run(test_data[:n_examples].eval())

    print("Predictions:", predictions)
    print("Actual     :", test_labels[:n_examples].eval().astype('int32'))

if __name__ == '__main__':
    main()