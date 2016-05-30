from __future__ import print_function
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from SaveCnn import CnnModel
from SaveCnn import HiddenLayer
from SaveCnn import LeNetConvPoolLayer
from SaveCnn import LogisticRegression


def predict():
    print("Load model.....")

    # load the saved model
    cnnModel = pickle.load(open('best_model.pkl'))
    aIndex = T.lscalar()

    # compile a predictor function
    # the parameter is works;

    predict_the_model = theano.function(
        inputs=[aIndex],
        outputs=theano.shared(cnnModel.predict_result),
        on_unused_input='ignore'
    )

    print("Start----------------------------------")
    predicted_values = predict_the_model(20)

    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    print("End----------------------------------")
"""
    Main method
"""
if __name__ == '__main__':
    predict()
