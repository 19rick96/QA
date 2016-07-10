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
word_vector_size = 200
word2vec = utils.load_glove(word_vector_size)
tr_num = 15000
res = 5
val_num = 1000

data = []

with open("super_data.csv") as tsvfile:
	tsvreader = csv.reader(tsvfile, delimiter = "\t")
	for line in tsvreader:
		data.append(line)

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

def process_input(data1):
	sents1 = []
	sents2 = []
	sim = []
	for i in range(1,len(data1)):
		s1 = data1[i][0]
		s2 = data1[i][1]
		score = float(data1[i][2])
		s1 = s1.strip()
		s1 = repl(s1)
		s1 = s1.lower().split(' ')
		s1 = [w for w in s1 if len(w) > 0]
		s1_vector = [utils.process_word(word = w, 
	                                word2vec = word2vec, 
	                                vocab = vocab, 
	                                ivocab = ivocab, 
	                                word_vector_size = word_vector_size, 
	                                to_return = "word2vec",silent=True) for w in s1]
		s2 = s2.strip()
		s2 = repl(s2)
		s2 = s2.lower().split(' ')
		s2 = [w for w in s2 if len(w) > 0]
		s2_vector = [utils.process_word(word = w, 
	                                word2vec = word2vec, 
	                                vocab = vocab, 
	                                ivocab = ivocab, 
	                                word_vector_size = word_vector_size, 
	                                to_return = "word2vec",silent=True) for w in s2]
		sents1.append(np.vstack(s1_vector))
		sents2.append(np.vstack(s2_vector))
		minn = 0.000001
		p = np.array([minn,minn,minn,minn,minn])
		if score == 5.0:
			p[4] = 1.0 - 4*minn
		else:
			p[int(np.floor(score))] = score - np.floor(score) + minn
			p[int(np.floor(score)) - 1] = 1 + np.floor(score) - score + minn
		sim.append(p)
	sents1 = np.asarray(sents1)
	sents2 = np.asarray(sents2)
	sim = np.asarray(sim)
	return sents1,sents2,sim

def extract_train(s1,s2,sim):
	s1_tr = s1[0:tr_num]
	s2_tr = s2[0:tr_num]
	sim_tr = sim[0:tr_num]
	return s1_tr,s2_tr,sim_tr

def extract_test(s1,s2,sim):
	s1_test = s1[tr_num:len(s1)]
	s2_test = s2[tr_num:len(s2)]
	sim_test = sim[tr_num:len(s1)]
	return s1_test,s2_test,sim_test

def constant_param(value=0.0, shape=(0,)):
	return theano.shared(init.Constant(value).sample(shape), borrow=True)
   
def normal_param(std=0.1, mean=0.0, shape=(0,)):
	return theano.shared(init.Normal(std, mean).sample(shape), borrow=True)

def uniform_param(shape=(0,4),func="tanh"):
	if func == "tanh":
		P = np.random.uniform(-np.sqrt(2.0/(shape[0]+shape[1])),np.sqrt(2.0/(shape[0]+shape[1])),(shape[0],shape[1]))
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

def pearson_score(p,q):
	score = (sum(p*q) - ((sum(p)*sum(q))/len(p)))/(np.sqrt((sum(p*p)-((sum(p)*sum(p))/len(p)))*(sum(q*q)-((sum(q)*sum(q))/len(q)))))
	return score

def gru_next_state(x_t,s_tm1,U0,W0,b0,U1,W1,b1,U2,W2,b2):
	z_t = T.nnet.hard_sigmoid(U0.dot(x_t) + W0.dot(s_tm1) + b0)
	r_t = T.nnet.hard_sigmoid(U1.dot(x_t) + W1.dot(s_tm1) + b1)
    	h_t = T.tanh(U2.dot(x_t) + W2.dot(s_tm1 * r_t) + b2)
   	s_t = ((T.ones_like(z_t) - z_t) * h_t) + (z_t * s_tm1)
	return s_t

def lstm_next_state(x_t,h_tm1,s_tm1,Ui,Wi,bi,Uf,Wf,bf,Uo,Wo,bo,Ug,Wg,bg):
	i = T.nnet.sigmoid(Ui.dot(x_t) + Wi.dot(s_tm1) + bi)
	f = T.nnet.sigmoid(Uf.dot(x_t) + Wf.dot(s_tm1) + bf)
	o = T.nnet.sigmoid(Uo.dot(x_t) + Wo.dot(s_tm1) + bo)
	g = T.tanh(Ug.dot(x_t) + Wg.dot(s_tm1) + bg)
	h_t = (h_tm1*f) + (g*i)
	s_t = T.tanh(h_t)*o
	return h_t,s_t

class DMN(object):
	def __init__(self,hid_dim,att_hid_dim,bptt_truncate = -1):
		self.hidden_dim = hid_dim
		self.bptt_truncate = bptt_truncate
		self.att_hid_dim = att_hid_dim
		"""
		# input lstm parameters
		self.Ui = normal_param(std=0.006, shape=(self.hidden_dim, word_vector_size))
		self.Uf = normal_param(std=0.006, shape=(self.hidden_dim, word_vector_size))
		self.Uo = normal_param(std=0.006, shape=(self.hidden_dim, word_vector_size))
		self.Ug = normal_param(std=0.006, shape=(self.hidden_dim, word_vector_size))
		self.Wi = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.Wf = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.Wo = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.Wg = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.bi = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.bf = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.bo = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.bg = constant_param(value=0.0, shape=(self.hidden_dim,))	
		"""
		#gru sentence parameters
		self.U0_i = normal_param(std=0.006, shape=(self.hidden_dim, word_vector_size))
		self.U1_i = normal_param(std=0.006, shape=(self.hidden_dim, word_vector_size))
		self.U2_i = normal_param(std=0.006, shape=(self.hidden_dim, word_vector_size))
		self.W0_i = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.W1_i = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.W2_i = normal_param(std=0.01, shape=(self.hidden_dim, self.hidden_dim))
		self.b0_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b1_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b2_i = constant_param(value=0.0, shape=(self.hidden_dim,))	
		#  1 attention mechanism parameters
		self.W1 = normal_param(std=0.0033, shape=(self.hidden_dim, (4*self.hidden_dim)+1))
		self.W2 = normal_param(std=0.01, shape=(res,self.hidden_dim))
		self.b1 = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b2 = constant_param(value=0.0, shape=(res,))
		self.Wb = normal_param(std=0.008, shape=(self.hidden_dim,self.hidden_dim))
		#  2 attention mechanism parameters
		self.Wx = normal_param(std=0.015, shape=(self.att_hid_dim, self.hidden_dim))
		self.Wd = normal_param(std=0.015, shape=(self.att_hid_dim, self.hidden_dim))
		self.b_h = constant_param(value=0.0, shape=(self.att_hid_dim,))
		self.b_p = constant_param(value=0.0, shape=(res,))
		self.Wp = normal_param(std=0.05, shape=(res,self.att_hid_dim))

		q = T.matrix()
		a = T.matrix()
		t = T.vector()

		s_a,a_updates = theano.scan(self.input_next_state,sequences=a,outputs_info=T.zeros_like(self.b0_i))
		s_q,q_updates = theano.scan(self.input_next_state,sequences=q,outputs_info=T.zeros_like(self.b0_i))

		q_q = s_q[-1]
		a_a = s_a[-1]	

		self.pred = self.attn_step_2(a_a,q_q)	

		self.loss = self.kl_div(t,self.pred)

		self.params = [self.U0_i,self.W0_i,self.b0_i,
				self.U1_i,self.W1_i,self.b1_i,
				self.U2_i,self.W2_i,self.b2_i,
				self.Wx,self.Wd,self.b_h,self.b_p,self.Wp]

		#self.loss = self.loss + 0.00003*l2_reg(self.params)
		updts = upd.adam(self.loss,self.params)

		self.train_fn = theano.function(inputs = [q,a,t], outputs = [self.pred,self.loss], updates = updts)
		self.test_fn = theano.function(inputs = [q,a], outputs = self.pred)
	
	def input_next_state(self,x_t,s_tm1):
		s_t = gru_next_state(x_t,s_tm1,self.U0_i,self.W0_i,self.b0_i,self.U1_i,self.W1_i,self.b1_i,self.U2_i,self.W2_i,self.b2_i)
		return s_t

	"""
	def lstm_inp_next_state(self,x_t,h_tm1,s_tm1):
		h_t,s_t = lstm_next_state(x_t,h_tm1,s_tm1,self.Ui,self.Wi,self.bi,self.Uf,self.Wf,self.bf,self.Uo,self.Wo,self.bo,
						self.Ug,self.Wg,self.bg)
		return h_t,s_t	
	"""
	def attn_step(self,a,q):
		aWq = T.stack([T.dot(T.dot(a, self.Wb), q)])
		z = T.concatenate([a,q,a*q,T.abs_(a-q),aWq],axis=0)
		l_1 = T.dot(self.W1, z) + self.b1
		l_1 = T.tanh(l_1)
		l_2 = T.dot(self.W2,l_1) + self.b2
		g = T.nnet.softmax(l_2)
		return g[0]

	def attn_step_2(self,l,r):
		dot = l*r
		diff = T.abs_(l-r) 
		l_h = T.nnet.sigmoid(T.dot(self.Wx,dot) + T.dot(self.Wd,diff) + self.b_h)
		l_p = T.nnet.softmax(T.dot(self.Wp,l_h) + self.b_p)
		return l_p[0]

	def kl_div(self,p,q):
		res = T.sum((p*T.log(p/q)),axis=0)
		return res
	
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
			#tr_q,tr_a,tr_t = shuffle(tr_q,tr_a,tr_t)
			for i in range(0,l):
				pred,loss = self.train_fn(tr_q[i],tr_a[i],tr_t[i])
				a_loss=a_loss+loss
				print "iteration : %d , %d" %((i+1),(j+1))
				print "loss : %.3f  average_loss : %.3f "%(loss,a_loss/(i+1))
				print "******************"
				if ((i+1)%10 == 0):
					fname = 'states_semnt/DMN2.epoch%d' %(j)
					self.save_params(fname,j)
s1_tr,s2_tr,sim_tr = process_input(data)
dmn = DMN(100,34)

s1_tr,s2_tr,sim_tr = shuffle(s1_tr,s2_tr,sim_tr)
s1_test,s2_test,sim_test = extract_test(s1_tr,s2_tr,sim_tr)
s1_train,s2_train,sim_train = extract_train(s1_tr,s2_tr,sim_tr)

for m in range(0,4):
	s1_train,s2_train,sim_train = shuffle(s1_train,s2_train,sim_train)
	s1_val = s1_train[0:val_num]
	s2_val = s2_train[0:val_num]
	sim_val = sim_train[0:val_num]
	s1_t = s1_train[val_num:tr_num]
	s2_t = s2_train[val_num:tr_num]
	sim_t = sim_train[val_num:tr_num]
	
	sc_i = 0
	for k in range(0,len(s1_val)):
		pr = dmn.test_fn(s1_val[k],s2_val[k])
		sc_i = sc_i + pearson_score(sim_val[k],pr)
	sc_i = sc_i/len(s1_val)

	dmn.train(s1_t,s2_t,sim_t,5)

	sc_f = 0
	for k in range(0,len(s1_val)):
		pr = dmn.test_fn(s1_val[k],s2_val[k])
		sc_f = sc_f + pearson_score(sim_val[k],pr)
	sc_f = sc_f/len(s1_val)
	file_specs = 'states_semnt/specs.txt'
	with open(file_specs, 'a') as f:
		f.write('initial pearson score : %f\n'%sc_i)
		f.write('final pearson score : %f\n'%sc_f)
		f.write('***************************************\n')
		f.write('\n')
	sc = 0
	for k in range(0,len(s1_test)):
		pr = dmn.test_fn(s1_test[k],s2_test[k])
		sc = sc + pearson_score(sim_test[k],pr)
	sc = sc/len(s1_test)
	file_specs = 'states_semnt/specs.txt'
	with open(file_specs, 'a') as f:
		f.write('TEST VALUE : ')
		f.write('pearson score : %f\n'%sc)
		f.write('***************************************\n')
		f.write('\n')
	print sc_i
	print sc_f
	print sc












