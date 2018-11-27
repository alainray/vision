import torch.nn as nn
import torch

class EntNet(nn.Module):
  def __init__(self,nDimensions=100,nMemoryNodes=20,seq_length=70, phi=nn.functional.prelu):
    super(EntNet, self).__init__()
    self.n_dimensions=nDimensions
    self.n_memory_nodes=nMemoryNodes
    self.seq_length=seq_length
    #Parameter matrices
    #Input Transformation
    self.F_i=nn.Parameter(torch.randn(self.n_dimensions, self.seq_length, dtype=torch.float))
    #Dynamic Memory 
    self.memory_nodes=torch.randn(self.n_dimensions, self.n_memory_nodes, dtype=torch.float)
    self.n_memory_nodes=nMemoryNodes
    self.keys=nn.Parameter(torch.randn(self.n_dimensions, self.n_memory_nodes, dtype=torch.float))
    self.U=nn.Parameter(torch.randn(self.n_dimensions, self.n_dimensions, dtype=torch.float))
    self.V=nn.Parameter(torch.randn(self.n_dimensions, self.n_dimensions, dtype=torch.float))
    self.W=nn.Parameter(torch.randn(self.n_dimensions, self.n_dimensions, dtype=torch.float))
    self.g=torch.randn(1, self.n_memory_nodes, dtype=torch.float)
    #Output Transformation
    self.F_q=nn.Parameter(torch.randn(self.n_dimensions, self.seq_length, dtype=torch.float))
    self.R=nn.Parameter(torch.randn(self.n_dimensions, self.n_dimensions, dtype=torch.float))
    self.H=nn.Parameter(torch.randn(self.n_dimensions, self.n_dimensions, dtype=torch.float))
    #Phi function
    self.phi=phi
    #prelu
    self.activation_parameter_mem=nn.Parameter(torch.randn(1,dtype=torch.float))
    self.activation_parameter_out=nn.Parameter(torch.randn(1,dtype=torch.float))

  
  def forward(self,input,query,memory=None):
    input=self.input_step(input,self.F_i)
    query=self.input_step(query,self.F_q)
    self.memory_nodes=self.memory_step(input)
    
    return self.output_step(query)
  
  def input_step(self, input, param):
    #Embedding of input vectors is assumed to be of size n_dimensions
    #Input to the EntNet is a sentence, always assumed to be of size 'seq_length'
    #If shorter, we pad with zero-vectors
    #If longer, we will ignore
    s=torch.zeros(1,self.n_dimensions,dtype=torch.float)
    length=len(input[0])
    for i, vector in enumerate(param.transpose(0,1)):
      if length > i: 
        s=s+vector*input[:,i] 
      else: 
        break
   
    return s.transpose(0,1)
  
  def memory_step(self, input):
    self.g=torch.sigmoid(torch.mm(input.transpose(0,1),self.memory_nodes)+torch.mm(input.transpose(0,1),self.keys))
    candidate_mem=torch.mm(self.U,self.memory_nodes)+torch.mm(self.V,self.keys)+torch.mm(self.W,input)
    candidate_mem=self.phi(candidate_mem,self.activation_parameter_mem)
    new_memory=torch.zeros(self.n_dimensions, self.n_memory_nodes, dtype=torch.float)
    for i,x in enumerate(self.g[0]):
      value=torch.mul(self.memory_nodes[:,i].clone(),(1+x))
      new_memory[:,i]=value
    return nn.functional.normalize(new_memory, p=2, dim=0)
    
  def output_step(self,query):
    p=torch.mm(query.transpose(0,1), self.memory_nodes)
    p=nn.functional.softmax(p,dim=1)
    u=torch.zeros(1,self.n_dimensions,dtype=torch.float)
    for e,p_i in enumerate(p[0]):
      u=u+p_i*self.memory_nodes[:,e]
    
    y=torch.mm(self.R,self.phi(query+torch.mm(self.H,u.transpose(0,1)),self.activation_parameter_out))
    
    return y
  def wipe_memory(self):
    self.memory_nodes=self.keys.clone()
