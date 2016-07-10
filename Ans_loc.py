import numpy as np
import theano
import theano.tensor as T
import os as os
import utils
import sys
import updates as upd
import init
import random
import cPickle as pickle
import csv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()

vocab = {}
ivocab = {}
word_vector_size = 50
word2vec = utils.load_glove(word_vector_size)

data_pairs = []
data_rand = []
q = []
a = []

with open("kalm.csv") as tsfile:
	tsreader = csv.reader(tsfile, delimiter = "\t")
	for line in tsreader:
		q.append(line)

with open("kalm_ans.csv") as tsvfile:
	tsvreader = csv.reader(tsvfile, delimiter = "\t")
	for line in tsvreader:
		a.append(line)

with open("dataset.csv") as tsvfile:
	tsvreader = csv.reader(tsvfile, delimiter = "\t")
	for line in tsvreader:
		data_pairs.append(line)

with open("rand_sentences.csv") as tsfile:
	tsreader = csv.reader(tsfile, delimiter = "\t")
	for line in tsreader:
		data_rand.append(line)

def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])

def dropout(X,p=0.):
	if p>0:
		retain_prob = 1-p
		X *= srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
		X /= retain_prob
	return X

def repl(q):
	q = q.replace('.',' . ')
	q = q.replace(',',' , ')
	q = q.replace('?',' ? ')
	q = q.replace(';',' ; ')
	q = q.replace(':',' : ')
	q = q.replace('"',' " ')
	q = q.replace('(',' ( ')
	q = q.replace(')',' ) ')
	q = q.replace('$',' $ ')
	q = q.replace('!',' ! ')
	q = q.replace('^',' ^ ')
	q = q.replace("-"," - ")
	q = q.replace('}',' } ')
	q = q.replace('{',' { ')
	q = q.replace('%',' % ')
	q = q.replace("'"," ' ")
	return q
	
def process_input_3(data1,data2):
	
	question = []
	answer = []

	for i in range(0,len(data1)):
		lines = data1[i]
		q = lines[0]
		q = q.strip()
		q = repl(q)
		q = q.lower().split(' ')
		q = [w for w in q if len(w) > 0]
		q_vector = [utils.process_word(word = w, 
	                                word2vec = word2vec, 
	                                vocab = vocab, 
	                                ivocab = ivocab, 
	                                word_vector_size = word_vector_size, 
	                                to_return = "word2vec",silent=True) for w in q]
		for j in range(0,len(data2)):
			a = data2[j][0]
			a = a.strip()
			a = repl(a)
			a = a.lower().split(' ')
			a = [w for w in a if len(w) > 0]
			a_vector = [utils.process_word(word = w, 
		                                word2vec = word2vec, 
		                                vocab = vocab, 
		                                ivocab = ivocab, 
		                                word_vector_size = word_vector_size, 
		                                to_return = "word2vec",silent=True) for w in a]
			question.append(np.vstack(q_vector))
			answer.append(np.vstack(a_vector))		
	
	question = np.asarray(question)
	answer = np.asarray(answer)
	print "processing data done ! ********************************"
	return question,answer

def process_input_2(data):
	questions1 = []
	answers1 = []
	truth1 = []
	questions0 = []
	answers0 = []
	truth0 = []
	questions = []
	answers = []
	truth = []

	for i in range(1,len(data)):
		lines = data[i]
		q = lines[1]
		a = lines[5]
		t = lines[6]
		if t == "0":
			truth0.append(0)
			q = q.strip()
			q = repl(q)
			q = q.lower().split(' ')
			q = [w for w in q if len(w) > 0]
			q_vector = [utils.process_word(word = w, 
		                                word2vec = word2vec, 
		                                vocab = vocab, 
		                                ivocab = ivocab, 
		                                word_vector_size = word_vector_size, 
		                                to_return = "word2vec",silent=True) for w in q]
			questions0.append(np.vstack(q_vector))
			a = a.strip()
			a = repl(a)
			a = a.lower().split(' ')
			a = [w for w in a if len(w) > 0]
			a_vector = [utils.process_word(word = w, 
		                                word2vec = word2vec, 
		                                vocab = vocab, 
		                                ivocab = ivocab, 
		                                word_vector_size = word_vector_size, 
		                                to_return = "word2vec",silent=True) for w in a]
			answers0.append(np.vstack(a_vector))
		if t == "1":
			truth1.append(1)
			q = q.strip()
			q = repl(q)
			q = q.lower().split(' ')
			q = [w for w in q if len(w) > 0]
			q_vector = [utils.process_word(word = w, 
		                                word2vec = word2vec, 
		                                vocab = vocab, 
		                                ivocab = ivocab, 
		                                word_vector_size = word_vector_size, 
		                                to_return = "word2vec",silent=True) for w in q]
			questions1.append(np.vstack(q_vector))
			a = a.strip()
			a = repl(a)
			a = a.lower().split(' ')
			a = [w for w in a if len(w) > 0]
			a_vector = [utils.process_word(word = w, 
		                                word2vec = word2vec, 
		                                vocab = vocab, 
		                                ivocab = ivocab, 
		                                word_vector_size = word_vector_size, 
		                                to_return = "word2vec",silent=True) for w in a]
			answers1.append(np.vstack(a_vector))
	questions1 = np.asarray(questions1)
	answers1 = np.asarray(answers1)
	truth1 = np.asarray(truth1)
	questions1,answers1,truth1 = shuffle(questions1,answers1,truth1)
	questions0 = np.asarray(questions0)
	answers0 = np.asarray(answers0)
	truth0 = np.asarray(truth0)
	questions0,answers0,truth0 = shuffle(questions0,answers0,truth0)
	for i in range(0,700):
		questions.append(questions1[i])
		answers.append(answers1[i])
		truth.append(truth1[i])
	for i in range(0,2100):
		questions.append(questions0[i])
		answers.append(answers0[i])
		truth.append(truth0[i])
	questions,answers,truth = shuffle(questions,answers,truth)
	questions,answers,truth = shuffle(questions,answers,truth)
	questions = np.asarray(questions)
	answers = np.asarray(answers)
	truth = np.asarray(truth)
	print "processing data done ! ********************************"
	return questions,answers,truth

def process_input(data1,data2):
	
	question = []
	answer = []
	truth = []
	cnt = 0
	combined = zip(data1,data2)
        random.shuffle(combined)
        data1,data2 = zip(*combined)
	for i in range(0,167):
		
		lines = data1[i]
		q = lines[0]
		a = lines[1]
		rand = random.sample(range(0, 260), 5)
		q = q.strip()
		q = repl(q)
		a = a.strip()
		a = repl(a)
		truth.append(1)
		q = q.lower().split(' ')
		q = [w for w in q if len(w) > 0]
		q_vector = [utils.process_word(word = w, 
	                                word2vec = word2vec, 
	                                vocab = vocab, 
	                                ivocab = ivocab, 
	                                word_vector_size = word_vector_size, 
	                                to_return = "word2vec",silent=True) for w in q]
		question.append(np.vstack(q_vector))
		a = a.lower().split(' ')
		a = [w for w in a if len(w) > 0]
		a_vector = [utils.process_word(word = w, 
	                                word2vec = word2vec, 
	                                vocab = vocab, 
	                                ivocab = ivocab, 
	                                word_vector_size = word_vector_size, 
	                                to_return = "word2vec",silent=True) for w in a]
		answer.append(np.vstack(a_vector))
		for j in range(0,5):
			a_rand = data2[rand[j]][0]
			a_rand = a_rand.strip()
			a_rand = repl(a_rand)
			a_rand = a_rand.lower().split(' ')
			a_rand = [w for w in a_rand if len(w) > 0]
			a_vector = [utils.process_word(word = w, 
			                        word2vec = word2vec, 
			                        vocab = vocab, 
			                        ivocab = ivocab, 
			                        word_vector_size = word_vector_size, 
			                        to_return = "word2vec",silent=True) for w in a_rand]
			if rand[j]==i:
				rand1 = random.sample(range(0, 260), 1)
				a_rand = data2[rand1[0]][0]
				a_rand = a_rand.strip()
				a_rand = repl(a_rand)
				a_rand = a_rand.lower().split(' ')
				a_rand = [w for w in a_rand if len(w) > 0]
				a_vector = [utils.process_word(word = w, 
					                word2vec = word2vec, 
					                vocab = vocab, 
					                ivocab = ivocab, 
					                word_vector_size = word_vector_size, 
					                to_return = "word2vec",silent=True) for w in a_rand]
				if rand1[0] == i:
					cnt = cnt + 1
			question.append(np.vstack(q_vector))
			answer.append(np.vstack(a_vector))
			truth.append(0)
	question = np.asarray(question)
	answer = np.asarray(answer)
	truth = np.asarray(truth)
	print "processing data done ! ********************************"
	return question,answer,truth

def constant_param(value=0.0, shape=(0,)):
    return theano.shared(init.Constant(value).sample(shape), borrow=True)
   
def normal_param(std=0.1, mean=0.0, shape=(0,)):
    return theano.shared(init.Normal(std, mean).sample(shape), borrow=True)

def uniform_param(shape=(0,4),func="tanh"):
	if func == "tanh":
		P = np.random.uniform(-np.sqrt(6/(shape[0]+shape[1])),np.sqrt(6/(shape[0]+shape[1])),(shape[0],shape[1]))
	else : 
		P = np.random.uniform(-4*np.sqrt(6/(shape[0]+shape[1])),4*np.sqrt(6/(shape[0]+shape[1])),(shape[0],shape[1]))
	P = theano.shared(value=P.astype(theano.config.floatX),borrow=True)
	return P

def shuffle(tr_q,tr_a,tr_t):
        print "==> Shuffling the train set"
        combined = zip(tr_q,tr_a,tr_t)
        random.shuffle(combined)
        tr_q,tr_a,tr_t = zip(*combined)
	return tr_q,tr_a,tr_t

def gru_next_state(x_t,s_tm1,U0,W0,b0,U1,W1,b1,U2,W2,b2):
	z_t = T.nnet.hard_sigmoid(U0.dot(x_t) + W0.dot(s_tm1) + b0)
	r_t = T.nnet.hard_sigmoid(U1.dot(x_t) + W1.dot(s_tm1) + b1)
    	h_t = T.tanh(U2.dot(x_t) + W2.dot(s_tm1 * r_t) + b2)
   	s_t = (T.ones_like(z_t) - z_t) * h_t + z_t * s_tm1
	return s_t

class DMN(object):
	def __init__(self,hid_dim,bptt_truncate = -1):
		self.hidden_dim = hid_dim
		self.bptt_truncate = bptt_truncate
		
		#gru sentence parameters
		self.U0_i = normal_param(std=0.01, shape=(self.hidden_dim, word_vector_size))
		self.U1_i = normal_param(std=0.01, shape=(self.hidden_dim, word_vector_size))
		self.U2_i = normal_param(std=0.01, shape=(self.hidden_dim, word_vector_size))
		self.W0_i = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.W1_i = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.W2_i = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.b0_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b1_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b2_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		
		#attention mechanism parameters
		self.W1 = normal_param(std=0.0033, shape=(2*self.hidden_dim, (4*self.hidden_dim)+1))
		self.W2 = normal_param(std=0.01, shape=(2,2*self.hidden_dim))
		self.b1 = constant_param(value=0.0, shape=(2*self.hidden_dim,))
		self.b2 = constant_param(value=0.0, shape=(2,))
		self.Wb = normal_param(std=0.01, shape=(self.hidden_dim,self.hidden_dim))		

		q = T.matrix()
		a = T.matrix()
		t = T.iscalar()
		
		q = dropout(q,0.08)		
		a = dropout(a,0.16)

		s_a,a_updates = theano.scan(self.input_next_state,sequences=a,outputs_info=T.zeros_like(self.b2_i))
		s_q,q_updates = theano.scan(self.input_next_state,sequences=q,outputs_info=T.zeros_like(self.b2_i))

		q_q = s_q[-1]
		a_a = s_a[-1]	

		self.pred = self.attn_step(a_a,q_q)	

		self.loss = T.mean(T.nnet.categorical_crossentropy(self.pred,T.stack([t])))

		self.params = [self.U0_i,self.W0_i,self.b0_i,
				self.U1_i,self.W1_i,self.b1_i,
				self.U2_i,self.W2_i,self.b2_i,
				self.W1,self.W2,self.b1,self.b2,self.Wb]

		self.loss = self.loss + 0.00007*l2_reg(self.params)
		updts = upd.adam(self.loss,self.params)

		self.train_fn = theano.function(inputs = [q,a,t], outputs = [self.pred,self.loss], updates = updts)
		self.test_fn = theano.function(inputs = [q,a], outputs = self.pred)

	def input_next_state(self,x_t,s_tm1):
		s_t = gru_next_state(x_t,s_tm1,self.U0_i,self.W0_i,self.b0_i,self.U1_i,self.W1_i,self.b1_i,self.U2_i,self.W2_i,self.b2_i)
		return s_t

	def attn_step(self,a,q):
		aWq = T.stack([T.dot(T.dot(a, self.Wb), q)])
		z = T.concatenate([a,q,a*q,T.abs_(a-q),aWq],axis=0)
		z = dropout(z,0.1)
		l_1 = T.dot(self.W1, z) + self.b1
		l_1 = T.tanh(l_1)
		l_1 = dropout(l_1,0.1)
		l_2 = T.dot(self.W2,l_1) + self.b2
		g = T.nnet.softmax(l_2)
		return g

	def save_params(self, file_name, epoch):
		with open(file_name, 'w') as save_file:
		    pickle.dump(
		        obj = {
		            'params' : [x.get_value() for x in self.params],
		            'epoch' : epoch, 
		        },
		        file = save_file,
		        protocol = -1
		    )
    
	def load_state(self, file_name):
		print "==> loading state %s" % file_name
		with open(file_name, 'r') as load_file:
			dict = pickle.load(load_file)
			loaded_params = dict['params']
			for (x, y) in zip(self.params, loaded_params):
				x.set_value(y)

	def train(self,tr_q,tr_a,tr_t,itr):
		l = len(tr_q)
		print "starting..."
		for j in range(0,itr):
			a_loss = 0.0
			tr_q,tr_a,tr_t = shuffle(tr_q,tr_a,tr_t)
			for i in range(0,l):
				pred,loss = self.train_fn(tr_q[i],tr_a[i],tr_t[i])
				a_loss=a_loss+loss
				print "iteration : %d , %d" %((i+1),(j+1))
				print "loss : %.3f  average_loss : %.3f  prediction : %.3f"%(loss,a_loss/(i+1),pred[0].argmax())
				print "******************"
				if ((i+1)%10 == 0):
					fname = 'states_ans_loc/DMN.epoch%d' %(j)
					self.save_params(fname,j)

dmn = DMN(60)
q_test,a_test = process_input_3(a,q)
"""
for m in range(0,4):
	q_train,a_train,t_train = process_input(data_pairs,data_rand)
	print len(q_train)

	pred_t = []
	for i in range(0,len(q_train)):
		pred = dmn.test_fn(q_train[i],a_train[i])
		pred_t.append(pred[0].argmax())
	actual_ones_i = sum(t_train)
	pred_ones_i = sum(pred_t)
	accuracy_i = sum([1 if t == p else 0 for t, p in zip(pred_t, t_train)])
	accuracy_ones_i = sum([1 if t == p == 1 else 0 for t, p in zip(pred_t, t_train)])
	print accuracy_i
	print actual_ones_i
	print pred_ones_i
	print accuracy_ones_i
	
	if m == 0 :
		iters = 12
	else :
		iters = 5
	dmn.train(q_train,a_train,t_train,iters)

	pred_t = []
	for i in range(0,len(q_train)):
		pred = dmn.test_fn(q_train[i],a_train[i])
		pred_t.append(pred[0].argmax())
	actual_ones_f = sum(t_train)
	pred_ones_f = sum(pred_t)
	accuracy_f = sum([1 if t == p else 0 for t, p in zip(pred_t, t_train)])
	accuracy_ones_f = sum([1 if t == p == 1 else 0 for t, p in zip(pred_t, t_train)])
	print accuracy_f
	print actual_ones_f
	print pred_ones_f
	print accuracy_ones_f

	file_specs = 'states_ans_loc/specs.txt'
	with open(file_specs, 'a') as f:
		f.write('accuracy initial : %f\n'%accuracy_i)
		f.write('actual_ones_i : %f\n'%actual_ones_i)
		f.write('pred_ones : %f\n'%pred_ones_i)
		f.write('accuracy of 1s : %f\n'%accuracy_ones_i)
		f.write('\n')
		f.write('accuracy final : %f\n'%accuracy_f)
		f.write('actual_ones_i : %f\n'%actual_ones_f)
		f.write('pred_ones : %f\n'%pred_ones_f)
		f.write('accuracy of 1s : %f\n'%accuracy_ones_f)
		f.write('***************************************\n')
		f.write('\n')
"""
for j in range(0,4):
	dmn.load_state("states_ans_loc/DMN.epoch%d"%j)
	pred_t = []
	for i in range(0,len(q_test)):
		l = []
		pred = dmn.test_fn(q_test[i],a_test[i])
		l.append(pred[0].argmax())
		pred_t.append(l)
	with open('result_p%d.csv'%j, 'w') as fp:
	    a = csv.writer(fp, delimiter='\t')
	    data = pred_t
	    a.writerows(data)


















