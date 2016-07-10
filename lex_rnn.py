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
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()
rng = np.random.RandomState(23455)

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
		p = (score - 1)/4.0
		sim.append(p)
	sents1 = np.asarray(sents1)
	sents2 = np.asarray(sents2)
	sim = np.asarray(sim)
	return sents1,sents2,sim

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

def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])

def dropout(X,p=0.):
	if p>0:
		retain_prob = 1-p
		X *= srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
		X /= retain_prob
	return X

def cos_sim(a,b):
	numer = sum(a*b)
	a_norm = np.sqrt(sum(a*a))
	b_norm = np.sqrt(sum(b*b))
	sim = numer/(a_norm*b_norm)
	return sim

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

def shuffle(tr_q,tr_a,tr_t):
        print "==> Shuffling the train set"
        combined = zip(tr_q,tr_a,tr_t)
        random.shuffle(combined)
        tr_q,tr_a,tr_t = zip(*combined)
	return tr_q,tr_a,tr_t

def pearson_score(p,q):
	score = (sum(p*q) - ((sum(p)*sum(q))/len(p)))/(np.sqrt((sum(p*p)-((sum(p)*sum(p))/len(p)))*(sum(q*q)-((sum(q)*sum(q))/len(q)))))
	return score

def mse(t,scr):
	return (t-scr)*(t-scr)

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

class LCD(object):
	def __init__(self,f_match,f_decomp,hidden_dim):
		fm = f_match.split("-")
		self.f_match = fm[0]
		self.fm_win = 0
		if len(fm) != 1:
			self.fm_win = int(fm[1])
		self.f_decomp = f_decomp
		self.hidden_dim = hidden_dim
		
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

		spl = T.matrix()
		smin = T.matrix()
		tpl = T.matrix()
		tmin = T.matrix()
		scr = T.scalar()
		
		s_plus,spl_updates = theano.scan(self.input_next_state,sequences=spl,outputs_info=T.zeros_like(self.b0_i))
		s_minus,smin_updates = theano.scan(self.input_next_state,sequences=smin,outputs_info=T.zeros_like(self.b0_i))
		t_plus,tpl_updates = theano.scan(self.input_next_state,sequences=tpl,outputs_info=T.zeros_like(self.b0_i))
		t_minus,tmin_updates = theano.scan(self.input_next_state,sequences=tmin,outputs_info=T.zeros_like(self.b0_i))

		s = T.concatenate([s_plus,s_minus],axis = 0)
		t = T.concatenate([t_plus,t_minus],axis=0)

		sc,sc_updates = theano.scan(self.l1,sequences = [s,t],outputs_info = None)
		self.score = T.exp(T.sum(sc))	

		self.loss = (scr-self.score)*(scr-self.score)

		self.params = [self.U0_i,self.W0_i,self.b0_i,
				self.U1_i,self.W1_i,self.b1_i,
				self.U2_i,self.W2_i,self.b2_i]

		#self.loss = self.loss + 0.00003*l2_reg(self.params)
		updts = upd.adam(self.loss,self.params)

		self.train_fn = theano.function(inputs = [spl,smin,tpl,tmin,scr], outputs = [self.score,self.loss], updates = updts)
		self.test_fn = theano.function(inputs = [spl,smin,tpl,tmin], outputs = self.score)
		#self.f = theano.function([s,t], [self.o_s,self.o_t])
	
	def input_next_state(self,x_t,s_tm1):
		s_t = gru_next_state(x_t,s_tm1,self.U0_i,self.W0_i,self.b0_i,self.U1_i,self.W1_i,self.b1_i,self.U2_i,self.W2_i,self.b2_i)
		return s_t

	"""
	def lstm_inp_next_state(self,x_t,h_tm1,s_tm1):
		h_t,s_t = lstm_next_state(x_t,h_tm1,s_tm1,self.Ui,self.Wi,self.bi,self.Uf,self.Wf,self.bf,self.Uo,self.Wo,self.bo,
						self.Ug,self.Wg,self.bg)
		return h_t,s_t	
	"""
	def decompose(self,s,t):
		#similarity matrix
		A = np.zeros((t.shape[0],s.shape[0]))
		for i in range(0,A.shape[0]):
			for j in range(0,A.shape[1]):
				A[i][j] = cos_sim(t[i],s[j])
		#creating semantic similarity matrix for each sentence (apply f_match) : global, local-w , max
		s_sem = []
		t_sem = []
		if self.f_match == "global":
			for i in range(0,len(s)):
				si = np.zeros(word_vector_size)
				for j in range(0,len(t)):
					si = si + (A[j][i]*t[j])
				s_sem.append(si)
			for i in range(0,len(t)):
				ti = np.zeros(word_vector_size)
				for j in range(0,len(s)):
					ti = ti + (A[i][j]*s[j])
				t_sem.append(ti)
		elif self.f_match == "local":
			At = A.T
			for i in range(0,len(s)):
				k = At[i].argmax()
				w = self.fm_win
				low = k-w
				high = k+w
				if low<0 : low = 0
				if high>len(t) : high = len(t)
				si = np.zeros(word_vector_size)
				for j in range(low,high):
					si = si + (A[j][i]*t[j])
				s_sem.append(si)
			for i in range(0,len(t)):
				k = A[i].argmax()
				w = self.fm_win
				low = k-w
				high = k+w
				if low<0 : low = 0
				if high>len(s) : high = len(s)
				ti = np.zeros(word_vector_size)
				for j in range(low,high):
					ti = ti + (A[i][j]*s[j])
				t_sem.append(ti)
		else:
			At = A.T
			for i in range(0,len(s)):
				arg = At[i].argmax()
				s_sem.append(t[arg])
			for i in range(0,len(t)):
				arg = A[i].argmax()
				t_sem.append(s[arg])
		s_sem = np.asarray(s_sem)
		t_sem = np.asarray(t_sem)
		#decomposition of similarity matrices  : rigid, linear, orthogonal
		splus = []
		sminus = []
		tplus = []
		tminus = []
		if self.f_decomp == "rigid":
			for i in range(0,len(s_sem)):
				if np.array_equal(s_sem[i],s[i]):
					splus.append(s[i])
					sminus.append(np.zeros(word_vector_size))
				else:
					sminus.append(s[i])
					splus.append(np.zeros(word_vector_size))
			for i in range(0,len(t_sem)):
				if np.array_equal(t_sem[i],t[i]):
					tplus.append(t[i])
					tminus.append(np.zeros(word_vector_size))
				else:
					tminus.append(t[i])
					tplus.append(np.zeros(word_vector_size))
		elif self.f_decomp == "linear":
			for i in range(0,len(s_sem)):
				alpha = cos_sim(s_sem[i],s[i])
				splus.append(alpha*s[i])
				sminus.append((1-alpha)*s[i])
			for i in range(0,len(t_sem)):
				alpha = cos_sim(t_sem[i],t[i])
				tplus.append(alpha*t[i])
				tminus.append((1-alpha)*t[i])
		else : 
			for i in range(0,len(s_sem)):
				coeff = sum(s_sem[i]*s[i])/sum(s_sem[i]*s_sem[i])
				splus.append(coeff*s_sem[i])
				sminus.append(s[i]-(coeff*s_sem[i]))
			for i in range(0,len(t_sem)):
				coeff = sum(t_sem[i]*t[i])/sum(t_sem[i]*t_sem[i])
				tplus.append(coeff*t_sem[i])
				tminus.append(t[i]-(coeff*t_sem[i]))
		splus = np.asarray(splus)
		sminus = np.asarray(sminus)
		tplus = np.asarray(tplus)
		tminus = np.asarray(tminus)
		return splus,sminus,tplus,tminus

	def l1(self,s1w,s2w):
		return -T.abs_(s1w-s2w)

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

	def train(self,s1,s2,sim_scr,itr):
		l = len(s1)
		print "starting..."
		for j in range(0,itr):
			a_loss = 0.0
			#tr_q,tr_a,tr_t = shuffle(tr_q,tr_a,tr_t)
			for i in range(0,l):
				sp,sm,tp,tm = self.decompose(s1[i],s2[i])
				pred,loss = self.train_fn(sp,sm,tp,tm,sim_scr[i])
				a_loss=a_loss+loss
				print "iteration : %d , %d" %((i+1),(j+1))
				print "loss : %.3f  average_loss : %.3f "%(loss,a_loss/(i+1))
				print "******************"
				if ((i+1)%10 == 0):
					fname = 'states_lex_comp_dec/lex_rnn.epoch%d' %(j)
					self.save_params(fname,j)

s1_tr,s2_tr,sim_tr = process_input(data)

lcd = LCD("max","orthogonal",50)

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
		sp,sm,tp,tm = lcd.decompose(s1_val[k],s2_val[k])
		pr = lcd.test_fn(sp,sm,tp,tm)
		sc_i = sc_i + mse(sim_val[k],pr)
	sc_i = sc_i/len(s1_val)
	sc_i = sc_i * 16.0

	lcd.train(s1_t,s2_t,sim_t,5)

	sc_f = 0
	for k in range(0,len(s1_val)):
		sp,sm,tp,tm = lcd.decompose(s1_val[k],s2_val[k])
		pr = lcd.test_fn(sp,sm,tp,tm)
		sc_f = sc_f + mse(sim_val[k],pr)
	sc_f = sc_f/len(s1_val)
	sc_f = sc_f * 16.0
	file_specs = 'states_lex_comp_dec/specs.txt'
	with open(file_specs, 'a') as f:
		f.write('initial mse score : %f\n'%sc_i)
		f.write('final mse score : %f\n'%sc_f)
		f.write('***************************************\n')
		f.write('\n')
	sc = 0
	for k in range(0,len(s1_test)):
		sp,sm,tp,tm = lcd.decompose(s1_test[k],s2_test[k])
		pr = lcd.test_fn(sp,sm,tp,tm)
		sc = sc + mse(sim_test[k],pr)
	sc = sc/len(s1_test)
	sc = sc * 16.0
	file_specs = 'states_lex_comp_dec/specs.txt'
	with open(file_specs, 'a') as f:
		f.write('TEST VALUE : ')
		f.write('mse score : %f\n'%sc)
		f.write('***************************************\n')
		f.write('\n')
	print sc_i
	print sc_f
	print sc

"""
s = np.array([[0.418 ,0.24968 ,-0.41242, 0.1217, 0.34527],
		[0.23682 ,-0.16899, 0.40951 ,0.63812 ,0.47709],
		[-0.17792, 0.42962, 0.032246, -0.41376, 0.13228]])
t = np.array([[-0.26854, 0.037334, -2.0932, 0.22171, -0.39868],
		[0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
		[0.418 ,0.24968 ,-0.41242 ,0.1217 ,0.34527],
		[-0.23938, 0.13001, -0.063734, -0.39575, -0.48162],
		[0.41705, 0.056763, -6.3681e-05, 0.068987, 0.087939]])
sf,tf,scre,ls = lcd.produce(s,t)
print sf
print tf
print "*********************************************************"
print sf.shape
print tf.shape
print scre
print ls
"""





















