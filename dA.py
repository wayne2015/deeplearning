import numpy
import theano
import theano.tensor as T
import cPickle,gzip
import utils
import Image
from theano.tensor.shared_randomstreams import RandomStreams

class dA(object):
    def __init__(self,input,n_visible,n_hidden,
                 w_init=None,bVisible=None,bHidden=None):
        numpy_rng = numpy.random.RandomState(123)
        #randomFunc = RandomStreams(numpy_rng.randint(2**30))
        if w_init  == None:
            w_init = numpy.asarray(
                numpy_rng.uniform(
                    size=(n_visible,n_hidden),
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible))
                ),
                dtype = theano.config.floatX
            )
        if bVisible == None:
            bVisible = theano.shared(
                value = numpy.zeros(
                    n_visible,
                    dtype = theano.config.floatX
                ),
                borrow = True
            )
        if bHidden == None:
            bHidden = theano.shared(
                value = numpy.zeros(
                        n_hidden,
                        dtype = theano.config.floatX
                ),
                borrow=True
            )
        if input == None:
            input = T.dmatrix(name='input')

        self.w = theano.shared(value=w_init,name='w',borrow=True)
        self.wT = self.w.T
        self.bVisible = bVisible
        self.bHidden = bHidden
        self.input = input
        self.params= (self.w,self.bVisible,self.bHidden)

    def GetHiddenValue(self,input):
        return T.nnet.sigmoid(T.dot(input,self.w)+self.bHidden)

    def GetReconstructedValue(self,selfOutput):
        return T.nnet.sigmoid(T.dot(selfOutput,self.wT)+self.bVisible)

    def CostFunction(self):
        corruptedData = self.GetCorrupted(self.input,level=0.2)
        selfOutput = self.GetHiddenValue(corruptedData)
        reStructedOutput = self.GetReconstructedValue(selfOutput)
        all_error = -T.sum(self.input*T.log(reStructedOutput)+(1-self.input)*T.log(1-reStructedOutput),axis=1)
        return T.mean(all_error)
    def GetCorrupted(self,input,level):
        numpy_rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2**30))
        return  theano_rng.binomial(size=input.shape, n=1, p=1 -level) * input
#######################################
##global function
def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

def load_data():
    f = gzip.open('mnist.pkl.gz','rb')
    train_set,valid_set,test_set = cPickle.load(f)
    f.close()
    train_x,train_y = shared_dataset(train_set)
    valid_x,valid_y = shared_dataset(valid_set)
    test_x,test_y = shared_dataset(test_set)
    return ((train_x,train_y),(valid_x,valid_y),(test_x,test_y))

if __name__ == '__main__':
    mydata = load_data()
    train_x,train_y = mydata[0]
    valid_x,valid_y = mydata[1]
    test_x,test_y = mydata[2]

    ##build model
    input = T.dmatrix("input")
    index =T.lscalar()
    learningRate = 0.1
    batchSize = 20
    nTrainBatch = train_x.get_value().shape[0]/batchSize
    epochs = 20
    dAcode = dA(input,28*28,500)
    cost =  dAcode.CostFunction()
    gparams = T.grad(cost,dAcode.params)
    updates = [(param,param-learningRate*gparam) for (param,gparam) in zip(dAcode.params,gparams)]

    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {input:train_x[index*batchSize:(index+1)*batchSize]}
    )
    for epoch in  xrange(epochs+1):
        for currentBatch in xrange(nTrainBatch):
            error = train_model(currentBatch)
            print "epoch %i, bacth %i/%i, cost %f "%(epoch, currentBatch+1,nTrainBatch,error)

    image = Image.fromarray(utils.tile_raster_images(X=dAcode.w.get_value(borrow=True).T,
                            img_shape=(28, 28), tile_shape=(10, 10),
                            tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')
