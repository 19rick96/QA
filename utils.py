import os as os
import numpy as np
import json
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from pprint import pprint
import sys
#from corenlp import *

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080),timeout=150000.0))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))

nlp = StanfordNLP()
components = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB","none","LOCATION","TIME","MONEY","PERSON","ORG","MONEY","PERCENT","DATE","NUMBER","none","which","what","whose","who","whom",
"what","where","when","how","why","none"]
def process_word2(sentence):
	dic = []
	result = nlp.parse(sentence)
	for j in range(0,len(result['sentences'])):
		sent = result['sentences'][j]['words']
		for i in range(0,len(sent)):
			vec = np.zeros(len(components))
			if sent[i][1]['PartOfSpeech'] in components:
				vec[components.index(sent[i][1]['PartOfSpeech'])] = 1
			else:
				vec[36] = 1
			if sent[i][1]['NamedEntityTag'] in components:
				vec[components.index(sent[i][1]['NamedEntityTag'])] = 1
			else:
				vec[46] = 1
			if sent[i][0].lower() in components:
				vec[components.index(sent[i][0].lower())] = 1
			else:
				vec[57] = 1
			dic.append(vec)
	dic = np.vstack(dic)
	return dic

def init_babi(fname):
    print "==> Loading test from %s" % fname
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "SF": ""} 
	    q_idx = []      
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:
            task["C"] += line
        else:
	    q_idx.append(id-1)
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
	    SF = tmp[2].strip()
	    SF = SF.split(" ")
	    SF = np.asarray(list((int(a)-1) for a in SF))
	    for i in reversed(q_idx):
	    	SF = [(a-1 if i<a else a) for a in SF]
	    SF = np.asarray(SF)
	    task["SF"] = SF
            tasks.append(task.copy())
    return tasks

def get_babi_raw(id, test_id):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    if (test_id == ""):
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/babi/en/%s_train.txt' % babi_name))
    babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/babi/en/%s_test.txt' % babi_test_name))
    print "done raw data *******************************"
    return babi_train_raw, babi_test_raw

            
def load_glove(dim):
    word2vec = {}
    
    print "==> loading glove"
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print "==> glove is loaded"
    
    return word2vec

def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.randint(0,99999,(word_vector_size,))/100000.0
    word2vec[word] = vector
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove.6B." + str(50) + "d.txt"),"a") as f1:
        string = word + ' ' + ' '.join(map(str,vector)) + "\n" 
	f1.write(string)
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector

def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")

def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)


import theano
import theano.tensor as T


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.
    Parameters
    ----------
    arr : array_like
        The data to be converted.
    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


def shared_empty(dim=2, dtype=None):
    """Creates empty Theano shared variable.
    Shortcut to create an empty Theano shared variable with
    the specified number of dimensions.
    Parameters
    ----------
    dim : int, optional
        The number of dimensions for the empty variable, defaults to 2.
    dtype : a numpy data-type, optional
        The desired dtype for the variable. Defaults to the Theano
        ``floatX`` dtype.
    Returns
    -------
    Theano shared variable
        An empty Theano shared variable of dtype ``dtype`` with
        `dim` dimensions.
    """
    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))


def as_theano_expression(input):
    """Wrap as Theano expression.
    Wraps the given input as a Theano constant if it is not
    a valid Theano expression already. Useful to transparently
    handle numpy arrays and Python scalars, for example.
    Parameters
    ----------
    input : number, numpy array or Theano expression
        Expression to be converted to a Theano constant.
    Returns
    -------
    Theano symbolic constant
        Theano constant version of `input`.
    """
    if isinstance(input, theano.gof.Variable):
        return input
    else:
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))


def collect_shared_vars(expressions):
    """Returns all shared variables the given expression(s) depend on.
    Parameters
    ----------
    expressions : Theano expression or iterable of Theano expressions
        The expressions to collect shared variables from.
    Returns
    -------
    list of Theano shared variables
        All shared variables the given expression(s) depend on, in fixed order
        (as found by a left-recursive depth-first search). If some expressions
        are shared variables themselves, they are included in the result.
    """
    # wrap single expression in list
    if isinstance(expressions, theano.Variable):
        expressions = [expressions]
    # return list of all shared variables
    return [v for v in theano.gof.graph.inputs(reversed(expressions))
            if isinstance(v, theano.compile.SharedVariable)]


def one_hot(x, m=None):
    """One-hot representation of integer vector.
    Given a vector of integers from 0 to m-1, returns a matrix
    with a one-hot representation, where each row corresponds
    to an element of x.
    Parameters
    ----------
    x : integer vector
        The integer vector to convert to a one-hot representation.
    m : int, optional
        The number of different columns for the one-hot representation. This
        needs to be strictly greater than the maximum value of `x`.
        Defaults to ``max(x) + 1``.
    Returns
    -------
    Theano tensor variable
        A Theano tensor variable of shape (``n``, `m`), where ``n`` is the
        length of `x`, with the one-hot representation of `x`.
    Notes
    -----
    If your integer vector represents target class memberships, and you wish to
    compute the cross-entropy between predictions and the target class
    memberships, then there is no need to use this function, since the function
    :func:`lasagne.objectives.categorical_crossentropy()` can compute the
    cross-entropy from the integer vector directly.
    """
    if m is None:
        m = T.cast(T.max(x) + 1, 'int32')

    return T.eye(m)[T.cast(x, 'int32')]


def unique(l):
    """Filters duplicates of iterable.
    Create a new list from l with duplicate entries removed,
    while preserving the original order.
    Parameters
    ----------
    l : iterable
        Input iterable to filter of duplicates.
    Returns
    -------
    list
        A list of elements of `l` without duplicates and in the same order.
    """
    new_list = []
    seen = set()
    for el in l:
        if el not in seen:
            new_list.append(el)
            seen.add(el)

    return new_list


def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).
    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements
    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.
    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


def compute_norms(array, norm_axes=None):
    """ Compute incoming weight vector norms.
    Parameters
    ----------
    array : numpy array or Theano expression
        Weight or bias.
    norm_axes : sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `array`. When this is not specified and `array` is a 2D array,
        this is set to `(0,)`. If `array` is a 3D, 4D or 5D array, it is
        set to a tuple listing all axes but axis 0. The former default is
        useful for working with dense layers, the latter is useful for 1D,
        2D and 3D convolutional layers.
        Finally, in case `array` is a vector, `norm_axes` is set to an empty
        tuple, and this function will simply return the absolute value for
        each element. This is useful when the function is applied to all
        parameters of the network, including the bias, without distinction.
        (Optional)
    Returns
    -------
    norms : 1D array or Theano vector (1D)
        1D array or Theano vector of incoming weight/bias vector norms.
    Examples
    --------
    >>> array = np.random.randn(100, 200)
    >>> norms = compute_norms(array)
    >>> norms.shape
    (200,)
    >>> norms = compute_norms(array, norm_axes=(1,))
    >>> norms.shape
    (100,)
    """

    # Check if supported type
    if not isinstance(array, theano.Variable) and \
       not isinstance(array, np.ndarray):
        raise RuntimeError(
            "Unsupported type {}. "
            "Only theano variables and numpy arrays "
            "are supported".format(type(array))
        )

    # Compute default axes to sum over
    ndim = array.ndim
    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 1:          # For Biases that are in 1d (e.g. b of DenseLayer)
        sum_over = ()
    elif ndim == 2:          # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}. "
            "Must specify `norm_axes`".format(array.ndim)
        )

    # Run numpy or Theano norm computation
    if isinstance(array, theano.Variable):
        # Apply theano version if it is a theano variable
        if len(sum_over) == 0:
            norms = T.abs_(array)   # abs if we have nothing to sum over
        else:
            norms = T.sqrt(T.sum(array**2, axis=sum_over))
    elif isinstance(array, np.ndarray):
        # Apply the numpy version if ndarray
        if len(sum_over) == 0:
            norms = abs(array)     # abs if we have nothing to sum over
        else:
            norms = np.sqrt(np.sum(array**2, axis=sum_over))

    return norms


def create_param(spec, shape, name=None):
    """
    Helper method to create Theano shared variables for layer parameters
    and to initialize them.
    Parameters
    ----------
    spec : numpy array, Theano expression, or callable
        Either of the following:
        * a numpy array with the initial parameter values
        * a Theano expression or shared variable representing the parameters
        * a function or callable that takes the desired shape of
          the parameter array as its single argument and returns
          a numpy array.
    shape : iterable of int
        a tuple or other iterable of integers representing the desired
        shape of the parameter array.
    name : string, optional
        If a new variable is created, the name to give to the parameter
        variable. This is ignored if `spec` is already a Theano expression
        or shared variable.
    Returns
    -------
    Theano shared variable or Theano expression
        A Theano shared variable or expression representing layer parameters.
        If a numpy array was provided, a shared variable is initialized to
        contain this array. If a shared variable or expression was provided,
        it is simply returned. If a callable was provided, it is called, and
        its output is used to initialize a shared variable.
    Notes
    -----
    This function is called by :meth:`Layer.add_param()` in the constructor
    of most :class:`Layer` subclasses. This enables those layers to
    support initialization with numpy arrays, existing Theano shared variables
    or expressions, and callables for generating initial parameter values.
    """
    shape = tuple(shape)  # convert to tuple if needed
    if any(d <= 0 for d in shape):
        raise ValueError((
            "Cannot create param with a non-positive shape dimension. "
            "Tried to create param with shape=%r, name=%r") % (shape, name))

    if isinstance(spec, theano.Variable):
        # We cannot check the shape here, Theano expressions (even shared
        # variables) do not have a fixed compile-time shape. We can check the
        # dimensionality though.
        # Note that we cannot assign a name here. We could assign to the
        # `name` attribute of the variable, but the user may have already
        # named the variable and we don't want to override this.
        if spec.ndim != len(shape):
            raise RuntimeError("parameter variable has %d dimensions, "
                               "should be %d" % (spec.ndim, len(shape)))
        return spec

    elif isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise RuntimeError("parameter array has shape %s, should be "
                               "%s" % (spec.shape, shape))
        return theano.shared(spec, name=name)

    elif hasattr(spec, '__call__'):
        arr = spec(shape)
        try:
            arr = floatX(arr)
        except Exception:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return an "
                               "array-like value")
        if arr.shape != shape:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return a value "
                               "with the correct shape")
        return theano.shared(arr, name=name)

    else:
        raise RuntimeError("cannot initialize parameters: 'spec' is not "
                           "a numpy array, a Theano expression, or a "
                           "callable")


def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps,
                go_backwards=False):
        """
        Helper function to unroll for loops. Can be used to unroll theano.scan.
        The parameter names are identical to theano.scan, please refer to here
        for more information.
        Note that this function does not support the truncate_gradient
        setting from theano.scan.
        Parameters
        ----------
        fn : function
            Function that defines calculations at each step.
        sequences : TensorVariable or list of TensorVariables
            List of TensorVariable with sequence data. The function iterates
            over the first dimension of each TensorVariable.
        outputs_info : list of TensorVariables
            List of tensors specifying the initial values for each recurrent
            value.
        non_sequences: list of TensorVariables
            List of theano.shared variables that are used in the step function.
        n_steps: int
            Number of steps to unroll.
        go_backwards: bool
            If true the recursion starts at sequences[-1] and iterates
            backwards.
        Returns
        -------
        List of TensorVariables. Each element in the list gives the recurrent
        values at each time step.
        """
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]

        # When backwards reverse the recursion direction
        counter = range(n_steps)
        if go_backwards:
            counter = counter[::-1]

        output = []
        prev_vals = outputs_info
        for i in counter:
            step_input = [s[i] for s in sequences] + prev_vals + non_sequences
            out_ = fn(*step_input)
            # The returned values from step can be either a TensorVariable,
            # a list, or a tuple.  Below, we force it to always be a list.
            if isinstance(out_, T.TensorVariable):
                out_ = [out_]
            if isinstance(out_, tuple):
                out_ = list(out_)
            output.append(out_)

            prev_vals = output[-1]

        # iterate over each scan output and convert it to same format as scan:
        # [[output11, output12,...output1n],
        # [output21, output22,...output2n],...]
        output_scan = []
        for i in range(len(output[0])):
            l = map(lambda x: x[i], output)
            output_scan.append(T.stack(*l))

        return output_scan
