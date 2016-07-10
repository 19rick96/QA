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

class LCD(object):
	def __init__(self,f_match,f_decomp,filter_no):
		fm = f_match.split("-")
		self.f_match = fm[0]
		self.fm_win = 0
		if len(fm) != 1:
			self.fm_win = int(fm[1])
		self.f_decomp = f_decomp
		self.fn = filter_no

		s = T.tensor4()
		t = T.tensor4()
		scr = T.scalar()

		w_shp1 = (self.fn,2,3,word_vector_size)
		w_shp2 = (self.fn,2,2,word_vector_size)
		w_shp3 = (self.fn,2,1,word_vector_size)
		w_bound1 = 2*3*word_vector_size
		w_bound2 = 2*2*word_vector_size
		w_bound3 = 2*1*word_vector_size
		b_shp = (self.fn,)
		self.W1 = theano.shared( np.asarray(np.random.uniform(
							low=-1.0 / w_bound1,
							high=1.0 / w_bound1,
							size=w_shp1),
						    dtype=s.dtype), name ='W1')
		self.W2 = theano.shared( np.asarray(np.random.uniform(
							low=-1.0 / w_bound2,
							high=1.0 / w_bound2,
							size=w_shp2),
						    dtype=s.dtype), name ='W2')
		self.W3 = theano.shared( np.asarray(np.random.uniform(
							low=-1.0 / w_bound3,
							high=1.0 / w_bound3,
							size=w_shp3),
						    dtype=s.dtype), name ='W3')
		self.b1 = theano.shared(np.asarray(
						    np.random.uniform(low=-.5, high=.5, size=b_shp),
						    dtype=s.dtype), name ='b1')
		self.b2 = theano.shared(np.asarray(
						    np.random.uniform(low=-.5, high=.5, size=b_shp),
						    dtype=s.dtype), name ='b2')
		self.b3 = theano.shared(np.asarray(
						    np.random.uniform(low=-.5, high=.5, size=b_shp),
						    dtype=s.dtype), name ='b3')

		conv_out_s1 = conv2d(s,self.W1)
		output_s1 = T.tanh(conv_out_s1 + self.b1.dimshuffle('x', 0, 'x', 'x'))
		output_s1 = output_s1.reshape((output_s1.shape[1],output_s1.shape[2]))
		o_s1,os1_updates = theano.scan(self.max_pool,sequences = output_s1,outputs_info = None)
		conv_out_s2 = conv2d(s,self.W2)
		output_s2 = T.tanh(conv_out_s2 + self.b2.dimshuffle('x', 0, 'x', 'x'))
		output_s2 = output_s2.reshape((output_s2.shape[1],output_s2.shape[2]))
		o_s2,os2_updates = theano.scan(self.max_pool,sequences = output_s2,outputs_info = None)
		conv_out_s3 = conv2d(s,self.W3)
		output_s3 = T.tanh(conv_out_s3 + self.b3.dimshuffle('x', 0, 'x', 'x'))
		output_s3 = output_s3.reshape((output_s3.shape[1],output_s3.shape[2]))
		o_s3,os3_updates = theano.scan(self.max_pool,sequences = output_s3,outputs_info = None)
		self.o_s = T.concatenate([o_s1,o_s2,o_s3],axis=0)

		conv_out_t1 = conv2d(t,self.W1)
		output_t1 = T.tanh(conv_out_t1 + self.b1.dimshuffle('x', 0, 'x', 'x'))
		output_t1 = output_t1.reshape((output_t1.shape[1],output_t1.shape[2]))
		o_t1,ot1_updates = theano.scan(self.max_pool,sequences = output_t1,outputs_info = None)
		conv_out_t2 = conv2d(t,self.W2)
		output_t2 = T.tanh(conv_out_t2 + self.b2.dimshuffle('x', 0, 'x', 'x'))
		output_t2 = output_t2.reshape((output_t2.shape[1],output_t2.shape[2]))
		o_t2,ot2_updates = theano.scan(self.max_pool,sequences = output_t2,outputs_info = None)
		conv_out_t3 = conv2d(t,self.W3)
		output_t3 = T.tanh(conv_out_t3 + self.b3.dimshuffle('x', 0, 'x', 'x'))
		output_t3 = output_t3.reshape((output_t3.shape[1],output_t3.shape[2]))
		o_t3,ot3_updates = theano.scan(self.max_pool,sequences = output_t3,outputs_info = None)
		self.o_t = T.concatenate([o_t1,o_t2,o_t3],axis=0)

		sc,sc_updates = theano.scan(self.l1,sequences = [self.o_s,self.o_t],outputs_info = None)
		self.score = T.exp(T.sum(sc))	

		self.loss = (scr-self.score)*(scr-self.score)

		self.params = [self.W1,self.W2,self.W3,self.b1,self.b2,self.b3]

		#self.loss = self.loss + 0.00003*l2_reg(self.params)
		updts = upd.adam(self.loss,self.params)

		self.train_fn = theano.function(inputs = [s,t,scr], outputs = [self.score,self.loss], updates = updts)
		self.test_fn = theano.function(inputs = [s,t], outputs = self.score)
		self.f = theano.function([s,t], [self.o_s,self.o_t])

	def max_pool(self,vec):
		maxp = vec[T.argmax(vec,axis=0)]
		return maxp
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
		s_final = []
		s_final.append(splus)
		s_final.append(sminus)
		s_final = np.asarray(s_final)
		t_final = []
		t_final.append(tplus)
		t_final.append(tminus)
		t_final = np.asarray(t_final)
		s_final = s_final.reshape(1,s_final.shape[0],s_final.shape[1],s_final.shape[2])
		t_final = t_final.reshape(1,t_final.shape[0],t_final.shape[1],t_final.shape[2])
		return s_final,t_final

	def l1(self,s1w,s2w):
		return -T.abs_(s1w-s2w)

	def produce(self,s,t):
		sd,td = self.decompose(s,t)
		s_vec,t_vec = self.f(sd,td)
		sc = self.test_fn(sd,td)
		return s_vec,t_vec,sc

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
				sd,td = self.decompose(s1[i],s2[i])
				pred,loss = self.train_fn(sd,td,sim_scr[i])
				a_loss=a_loss+loss
				print "iteration : %d , %d" %((i+1),(j+1))
				print "loss : %.3f  average_loss : %.3f "%(loss,a_loss/(i+1))
				print "******************"
				if ((i+1)%10 == 0):
					fname = 'states_lex_comp_dec/lex.epoch%d' %(j)
					self.save_params(fname,j)

s1_tr,s2_tr,sim_tr = process_input(data)

lcd = LCD("max","orthogonal",500)

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
		sd,td = lcd.decompose(s1_val[k],s2_val[k])
		pr = lcd.test_fn(sd,td)
		sc_i = sc_i + mse(sim_val[k],pr)
	sc_i = sc_i/len(s1_val)
	sc_i = sc_i * 16.0

	lcd.train(s1_t,s2_t,sim_t,5)

	sc_f = 0
	for k in range(0,len(s1_val)):
		sd,td = lcd.decompose(s1_val[k],s2_val[k])
		pr = lcd.test_fn(sd,td)
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
		sd,td = lcd.decompose(s1_test[k],s2_test[k])
		pr = lcd.test_fn(sd,td)
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





















