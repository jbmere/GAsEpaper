#!/usr/bin/python
__author__ = 'JOM'
 
import sys,getopt,csv,random,array,multiprocessing
from deap import base, creator, tools,algorithms

def evaluate(individual):
   print(individual), sum(individual)
   return sum(individual),

creator.create("FitnessMin",base.Fitness,weights=(-1.0,)) 
creator.create("Individual",list,typecode='d',fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def main():
   random.seed(64)
   i = 0
   CXPB , MUTPB = 0.35, 0.08
   total = len(sys.argv)
   myopts, args = getopt.getopt(sys.argv[1:],"p:n:s:g:")
   #
   # print ("Call: %s " % str(myopts))
   # o == option
   # a == argument for the option o
   #
   for o, a in myopts:
      if o == '-p':
         plen = int(a)
      elif o == '-s':
         size = int(a)
      elif o == '-m':
         mx = int(a)
      elif o == '-n':
         niter = int(a)
      else:
         print("Usage: %s -p psize -s isize -o MAX -n Niter" % sys.argv[0])
      
   toolbox.register("attr_int",random.randint,2,8)
   toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_int, size)
   toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   # Operator registering
   toolbox.register("evaluate", evaluate)
   toolbox.register("mate", tools.cxBlend,0.3)
   toolbox.register("mutate", tools.mutUniformInt,2,8, indpb=0.05)
   toolbox.register("select", tools.selTournament, tournsize=3)
   #
   ind1 = toolbox.individual()
   ind1.fitness.values = evaluate(ind1)
   print ind1
   mutant = toolbox.clone(ind1)
   ind2, = tools.mutGaussian(mutant,mu=0.0,sigma=0.2,indpb=0.2)
   del mutant.fitness.values
   #
   pop = toolbox.population(n=plen)
   for g in range(plen):
        print "-- Generation %i --" % g
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                tools.cxBlend(child1, child2,0.5)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                tools.mutGaussian(mutant,mu=0.0,sigma=0.2,indpb=0.2)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
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
