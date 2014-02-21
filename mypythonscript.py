#! /usr/bin/python
from pycallgraph import PyCallGraph
from pycallgraph.output  import  GraphvizOutput

with PyCallGraph(output=GraphvizOutput()):
   main()
