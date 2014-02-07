#!/usr/bin/python
__author__ = 'JOM'
 
import sys,getopt,csv,random,array,re,numpy,copy,math,functools
from deap import base, creator, tools,algorithms

rmpa={'D':0,'I':1,'P':2}
mpa ={0:'D',1:'I',2:'P'}
gcst = {}
gseq = {}
gr   = []
# Thanks for the path-finder to http://stackoverflow.com/questions/713508/find-the-paths-between-two-given-nodes
class MyQUEUE: # just an implementation of a queue
    def __init__(self):
    	self.holder = []
    def enqueue(self,val):
    	self.holder.append(val)
    def dequeue(self):
    	val = None
    	try:
    		val = self.holder[0]
    		if len(self.holder) == 1:
    			self.holder = []
    		else:
    			self.holder = self.holder[1:]	
    	except:
    		pass
    	return val	
    def IsEmpty(self):
    	result = False
    	if len(self.holder) == 0:
    		result = True
    	return result


def BFS(graph,start,end,q):
    temp_path = [start]
    q.enqueue(temp_path)
    res=[]
    while q.IsEmpty() == False:
    	tmp_path = q.dequeue()
    	last_node = tmp_path[len(tmp_path)-1]
    	if last_node == end:
    		res.append(tmp_path)
    	for link_node in graph[last_node]:
    		if link_node not in tmp_path:
    			#new_path = []
    			new_path = tmp_path + [link_node]
    			q.enqueue(new_path)
    return(res)

def graph(trns):
   seq=range(len(mpa))
   gr = {}
   for i in seq:
      gr[mpa[i]] = [ mpa[j] for j in seq if trns[i][j] > 0 ]
   return(gr)

def bestpath(ruta,coste,trns):
   ctr = 0
   st  = len(coste)*[None]
   ost = copy.copy(st)
   if len(ruta) == 1:
      st  = len(coste)*[ruta[0]]
      ctr = sum(numpy.asarray(coste) * trns[0][rmpa[ruta[0]]][rmpa[ruta[0]]])
      return (ctr,st)
   ns  = len(ruta)-1
   dtr = 0
   seq = len(coste)*[None]
   pot = copy.copy(seq)
   for i in range(ns):
      dtr += trns[1][rmpa[ruta[i]]][rmpa[ruta[i+1]]]
   lstp= len(coste) - dtr # lstp pasos para distribuir
   mctr= 10000000.
   if lstp < 0:
      return (mctr,ost)
   for i in range(lstp+1):
      st  = len(coste)*[None]
      rrta= copy.copy(ruta)
      ctr = sum(numpy.asarray(coste[0:i])*trns[0][rmpa[ruta[0]]][rmpa[ruta[0]]])
      st[0:i] = i*[ruta[0]]
      dt=trns[1][rmpa[ruta[0]]][rmpa[ruta[1]]]
      et=trns[0][rmpa[ruta[0]]][rmpa[ruta[1]]]
      ctr += sum(numpy.asarray(coste[i:i+dt])*et)
      st[i:i+dt] = dt * [''.join(ruta[0:2])]
      rrta.pop(0)
      rctr,rst = bestpath(rrta,coste[i+dt:],trns)
      ctr += rctr
      st[i+dt:] = rst
      if ( mctr > ctr ):
         mctr=ctr
         ost = st
   return(mctr,ost)
         
def eval_seg(frm,to,pos,cs,ind,cost,dj,trns,fvp,gr):
   path_queue = MyQUEUE()
   tc = 0
   lfe= len(cost) - cs
   lgp= lfe
   if pos >= 0:
      fi = ind[2][pos]
      dt = int(math.ceil(dj[ind[1][pos]]/fvp[0][fi]))  # Duracion del job
      lfe= ind[0][pos]+dt
      lgp= ind[0][pos]
      et = trns[0][len(trns)][len(trns)]*fvp[1][fi]
   fe    = lgp * [None]
   spath = lgp * [None]
   if frm == to:
      rutas=[[frm]]
      for i in range(len(mpa)):
          if frm != mpa[i]:
             ruta1=BFS(gr,frm,mpa[i],path_queue)
             ruta2=BFS(gr,mpa[i],to,path_queue)
             for r1 in ruta1:
                for r2 in ruta2:
                   ddl = r2.pop(0)
                   rt=r1+r2
                   if rt not in rutas:
                      rutas.append(rt)
   else:
      rutas=BFS(gr,frm,to,path_queue)
   mc=10000000.
   for ir in rutas:  # Ruta de menos coste para 'P'
      cst,op = bestpath(ir,cost[cs:cs+lgp],trns)
      if ( cst < mc ):
         mc=cst
         spath=op
   tc += mc
   fe[0:len(spath)]=spath
   cs += len(spath)
   return(len(fe),tc,fe)

def calcula(ind,i,pos,cost,dj,trns,fvp):
   fi = ind[2][i]
   dt = int(math.ceil(dj[ind[1][i]]/fvp[0][fi]))  # Duracion del job
   et = trns[0][len(trns)][len(trns)]*fvp[1][fi]
   tc = sum(numpy.asarray(cost[pos:pos+dt])*et)
   return (dt,tc)

def evaluate(ind,cost,dj,trns,fvp):
   global gcst, gseq
   fe = len(cost)*[None]
   nc = len(ind[0])
   dt = 0.
   for pos in range(nc):
      dt += math.ceil(dj[ind[1][pos]]/fvp[0][ind[2][pos]])
   fac= int(math.ceil(sum(ind[0])/(len(cost)-dt)))
   if fac > 1:
      ind[0] = [ int(math.ceil(x/fac)) for x in ind[0] ]
   dtf= sum(ind[0])
   if dt+dtf > len(cost):
      return (10000000.),
   tc,cs = 0,0
   fe[cs]='D'  #  Comenzamos en Down
   # gr=graph(trns[1])
   for i in range(nc):
      frm = fe[cs]
      to  = 'P'
      lng = ind[0][i]
      idx = (frm,to,cs,lng)
      if idx in gcst:
         tc += gcst[idx]
         fe[cs:cs+len(gseq[idx])] = gseq[idx]
         cs += len(gseq[idx])
      else:
         ncs,ptc,nsg   = eval_seg(frm,to,i,cs,ind,cost,dj,trns,fvp,gr)
         gcst[idx]     = ptc
         gseq[idx]     = nsg
         fe[cs:cs+ncs] = nsg
         cs += ncs 
         tc += ptc
      if cs >= len(cost):
         return (10000000.),
      fe[cs] = 'P'
      idt,itc=calcula(ind,i,pos,cost,dj,trns,fvp)
      if cs+idt >= len(cost):
         return (10000000.),
      fe[cs:cs+idt] = idt * ['P']
      tc += itc
      cs += idt
      fe[cs] = 'P'
   idx = (fe[cs],'D',cs,len(cost)-cs)
   if idx in gcst:
      tc += gcst[idx]
      fe[cs:len(cost)] = gseq[idx]
      cs = len(cost)
   else:
      ncs,ptc,nsg = eval_seg(fe[cs],'D',-1,cs,ind,cost,dj,trns,fvp,gr)
      gcst[idx]     = ptc
      gseq[idx]     = nsg
      fe[cs:len(cost)] = nsg
      tc += ptc
      cs = len(cost)
   return (tc),

def checkBounds(T,dj,fvp):
    def decorator(func):
        def wrappper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                fv = fvp[0]
		fvi= [fv[j] for j in child[2]]
		dji= [dj[j] for j in child[1]]
                dur=  [math.ceil(int(d)/float(f)) for d,f in zip (dji,fvi)]
		rem= T - sum(dur) - 2
		tw = sum(child[0])
 		if ( tw > rem ):
                   for i in range(len(child[0])):
                      child[0][i] = math.round(child[0][i]*rem/tw)

            return offspring
        return wrappper
    return decorator

def my_split(s, seps):
    res = [s]
    for sep in seps:
        s, res = res, []
        for seq in s:
	    wc = seq.split(sep)
	    if not (len(wc) == 1 and len(wc[0]) == 0):
                res += wc
    return res

def leerlineacsv(df):
   while (1):
      line = df.readline()
      line = line.strip(' \t\n')
      line = re.sub('#.*$','',line)
      newl = my_split(line,[' ',',',';',':'])
      if len(newl) == 1:
	    return(newl[0])
      if len(newl) > 1:
	    return(newl)

def crossover(ind1,ind2,alpha):
    for i,(x1,x2) in enumerate(zip(ind1[0],ind2[0])):
        gamma = abs((1. + 2 * alpha) * random.random() - alpha)
        if gamma > 1.:
           gamma = gamma-1
        ind1[0][i] = int( math.floor((1. - gamma) * x1 + gamma * x2))
        ind2[0][i] = int( math.floor(gamma * x1 + (1. - gamma) * x2))

    ind1[1],ind2[1] = tools.cxOrdered(ind1[1],ind2[1])
    pos = int(math.floor(len(ind1[2]) * random.random() ))
    for i,(x1,x2) in enumerate(zip(ind1[2],ind2[2])):
        if i <= pos:
           a = ind2[2][i]
           ind2[2][i] = ind1[2][i]
           ind1[2][i] = a
   
    return ind1, ind2

def mutation(ind,mu,sigma,pr,ll,ul):
    for i in xrange(len(ind[0])):
       if random.random() < pr:
          ind[0][i] += int(math.ceil(random.gauss(mu,sigma)))
          if ind[0][i] < 0:
             ind[0][i] = 0
    if random.random() < pr:
       ind[1] = tools.mutShuffleIndexes(ind[1],pr)[0]
    for i in xrange(len(ind[2])):
       if random.random() < pr:
          ind[2][i] = random.randint(ll,ul)

    return ind,

def main():
   global gr
   # random.seed(64)
   hist = tools.History()
   i = 0
   ifile = ''
   ofile = ''
   pfile = ''
   total = len(sys.argv)
   myopts, args = getopt.getopt(sys.argv[1:],"p:n:s:c:m:i:o:")
   #
   # print ("Call: %s " % str(myopts))
   # o == option
   # a == argument for the option o
   for o, a in myopts:
      if o == '-s':
         size = int(a)
      elif o == '-c':
         pcrs = float(a)
      elif o == '-m':
         pmut = float(a)
      elif o == '-n':
         niter = int(a)
      elif o == '-p':
         pfile = a
      elif o == '-i':
         ifile = a
      else:
         print("Usage: %s -L psize -s isize -c pcross -m pmut -n Niter -i ifile " % sys.argv[0])
      
   # Comenzamos a leer el csv
   f = open ( ifile, 'r')
   njobs = int (leerlineacsv(f))    # Numero de jobs a procesar
   lista = leerlineacsv(f)
   if len(lista) < njobs:
	print "No hay componentes suficientes DJ ",lista
   else:
	dj=[int(x) for x in lista]  # Duracion nominal de cada job

   T = int(leerlineacsv(f))          # Leemos el total de pasos de tiempo
   lista = leerlineacsv(f)
   if len(lista) < T:
	print "No hay componentes suficientes Et ",lista
   else:
	et=[float(x) for x in lista] # Leemos el coste en cada paso de tiempo

   nest = int (leerlineacsv(f))    # Leemos el numero de estados
   mce=[]
   for i in range(nest):
    	lista = leerlineacsv(f)
    	if len(lista) < nest:
	   print "No hay componentes suficientes Est ",lista
        else:
	   mce.append([float(x) for x in lista])

   mcd=[]
   for i in range(nest):
    	lista = leerlineacsv(f)
    	if len(lista) < nest:
	   print "No hay componentes suficientes Est ",lista
        else:
	   mcd.append([int(x) for x in lista])

   trns = [mce,mcd]  # Transiciones: Mat energia/pasoT + Mat duracion
   gr=graph(mcd) # GR sera usado en evaluaciones
   nv=int(leerlineacsv(f))   # Leemos el numero de factores de velocidad
   lista = leerlineacsv(f)
   if len(lista) < nv:
	print "No hay componentes suficientes Fv ",lista
   else:
	fv=[float(x) for x in lista]

   lista = leerlineacsv(f)
   if len(lista) < nv:
	print "No hay componentes suficientes Fp ",lista
   else:
	fp=[float(x) for x in lista]
    
   fvp = [fv,fp]
   # print njobs, T, nest, nv
   # print "DJ:",dj
   # print "T :",et
   print "Margen: ",T-sum(dj)
   #
   creator.create("FitnessMin",base.Fitness,weights=(-1.0,)) 
   creator.create("Individual",list,fitness=creator.FitnessMin)
   toolbox = base.Toolbox()
   #
   toolbox.register("attr_wait",random.randint,0,(T-sum(dj)-1))
   toolbox.register("indices",random.sample,range(njobs),njobs)
   toolbox.register("attr_pow",random.randint,0,nv-1)
   toolbox.register("waiting",tools.initRepeat, list,toolbox.attr_wait,njobs)
   toolbox.register("speed",tools.initRepeat, list,toolbox.attr_pow,njobs)
   toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.waiting,toolbox.indices,toolbox.speed), 1)
   toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   # Operator registering
   toolbox.register("evaluate", evaluate,et,dj,trns,fvp)
   toolbox.register("mate", crossover,pcrs)
   toolbox.register("mutate", tools.mutUniformInt,0,(T-sum(dj)), indpb=pmut)
   toolbox.register("select", tools.selTournament, tournsize=3)
   toolbox.decorate("mate", checkBounds(T,dj,fvp))
   toolbox.decorate("mutate", checkBounds(T,dj,fvp))
   #
   pop = toolbox.population(n=size)
   fitnesses = map(functools.partial(evaluate, cost=et,dj=dj,trns=trns,fvp=fvp),pop)
   for ind, fit in zip(pop, fitnesses):
       ind.fitness.values = fit
   #
   for g in range(niter):
        print "-- Generation %i --" % g
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < pcrs:
               crossover(child1, child2, pcrs)
               del child1.fitness.values
               del child2.fitness.values

        for mutant in offspring:
            if random.random() < pmut:
               mutation(mutant,mu=0.0,sigma=5.,pr=pmut,ll=0,ul=nv-1)
               del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(functools.partial(evaluate, cost=et,dj=dj,trns=trns,fvp=fvp),invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print "  Evaluated %i individuals" % len(invalid_ind)
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print "  Min %s" % min(fits)
        print "  Max %s" % max(fits)
        print "  Avg %s" % mean
        print "  Std %s" % std
    
   print "-- End of (successful) evolution --"
   best_ind = tools.selBest(pop, 1)[0]
   print "Best individual is %s, %s" % (best_ind, best_ind.fitness.values)

if __name__ == "__main__":
    main()
