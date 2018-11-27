import torch.nn as nn
import torch
from entnet import EntNet

class RelNet(EntNet):
    def __init__(self,
                 nDimensions=100,
                 nMemoryNodes=20,
                 seq_length=70,
                 phi=nn.functional.prelu):
        EntNet.__init__(self,nDimensions,nMemoryNodes,seq_length,phi)        
        #Relational Memory
        self.activation_parameter_rel_mem = nn.Parameter(
            torch.randn(1, dtype=torch.float))
        self.relational_memory = torch.randn(
            self.n_memory_nodes,
            self.n_memory_nodes,
            self.n_dimensions,
            1,
            dtype=torch.float)
        self.A = nn.Parameter(
            torch.randn(
                self.n_dimensions, self.n_dimensions, dtype=torch.float))
        self.B = nn.Parameter(
            torch.randn(
                self.n_dimensions, self.n_dimensions, dtype=torch.float))
        self.C = nn.Parameter(
            torch.randn(
                self.n_dimensions, 3 * self.n_dimensions, dtype=torch.float))

    def forward(self, input, query):

        input = self.input_step(input, self.F_i)
        query = self.input_step(query, self.F_q)
        self.memory_step(input)
        self.relational_step(input)

        return self.output_step(query)

    def relational_step(self, input):
        g_r = torch.zeros(
            self.n_memory_nodes,
            self.n_memory_nodes,
            self.n_dimensions,
            1,
            dtype=torch.float)
        new_input = input.transpose(0, 1)

        for i in range(0, self.n_memory_nodes):
            for j in range(0, self.n_memory_nodes):
                g_r[i][j] = self.g[0][i] * self.g[0][j] * torch.sigmoid(
                    torch.mm(new_input, self.relational_memory[i][j]))

        temp_input = torch.mm(self.B, input)
        candidate_mem = torch.randn(
            self.n_memory_nodes,
            self.n_memory_nodes,
            self.n_dimensions,
            1,
            dtype=torch.float)

        for i in range(0, self.n_memory_nodes):
            for j in range(0, self.n_memory_nodes):
                value = torch.mm(self.A,
                                 self.relational_memory[i][j]) + temp_input
                candidate_mem[i][j] = value

        candidate_mem = self.phi(candidate_mem,
                                 self.activation_parameter_rel_mem)
        new_relational_memory=torch.randn(
            self.n_memory_nodes,
            self.n_memory_nodes,
            self.n_dimensions,
            1,
            dtype=torch.float)
        for i in range(0, self.n_memory_nodes):
            for j in range(0, self.n_memory_nodes):
                new_relational_memory[i][j] =self.relational_memory[i][j]+ g_r[i][j] * candidate_mem[i][j]

        self.relational_memory = nn.functional.normalize(
            new_relational_memory, p=2, dim=2)

    def output_step(self, query):
        candidate_output = torch.zeros(
            self.n_memory_nodes,
            self.n_memory_nodes,
            self.n_dimensions,
            1,
            dtype=torch.float)
        p_ij = torch.zeros(
            self.n_memory_nodes, self.n_memory_nodes, 1, dtype=torch.float)
        
        for i in range(0, self.n_memory_nodes):
            for j in range(0, self.n_memory_nodes):
                mem_temp = torch.cat(
                    (self.memory_nodes[:, i], self.memory_nodes[:, j],
                     self.relational_memory[i][j].view(self.n_dimensions)), 0)
                candidate_output[i][j] = torch.mm(self.C,
                                                  mem_temp.view(
                                                      3 * self.n_dimensions,
                                                      1))
        for i in range(0, self.n_memory_nodes):
          for j in range(0, self.n_memory_nodes):       
            p_ij[i][j] = torch.mm(
            query.transpose(0, 1), candidate_output[i][j])

        new_p_ij = nn.functional.softmax(
            p_ij.view(self.n_memory_nodes * self.n_memory_nodes), dim=0).view(
                self.n_memory_nodes, self.n_memory_nodes, 1)

        u = torch.zeros(self.n_dimensions, 1, dtype=torch.float)

        for i in range(0, self.n_memory_nodes):
            for j in range(0, self.n_memory_nodes):
                u = u + new_p_ij[i][j] * candidate_output[i][j]
        y = torch.mm(self.R,
                     self.phi(query + torch.mm(self.H, u),
                              self.activation_parameter_out))
        return y
      
    def wipe_memory(self):
      self.memory_nodes=self.keys.clone()
      self.relational_memory = torch.randn(
            self.n_memory_nodes,
            self.n_memory_nodes,
            self.n_dimensions,
            1,
            dtype=torch.float)
      #This is probably incorrect, I'll just
      #do an interpolation of the related key values
      for i in range(0, self.n_memory_nodes):
        for j in range(0, self.n_memory_nodes):
          #self.relational_memory[i][j]=(self.keys[:,i].view(self.n_dimensions,1))
          self.relational_memory[i][j]=(self.keys[:,i].view(self.n_dimensions,1)+self.keys[:,j].view(self.n_dimensions,1))/2
            
