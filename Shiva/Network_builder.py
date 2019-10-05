import subprocess

class NetworkBuilder ():
	def __init__ (self, input_size, output_size, configs):

		if configs["algorithm"] == "DQN":
			
			# make a copy of dqnettemplate and call it fileName.py
			subprocess.call("cp /Network_Templates/DQNetTemplate.py DQNet1.py")
			
			# use the configs to append the rest of the network to fileName.py
			with open("DQNet1.py", "a") as file:
				file.write("\t\tself.net = nn.Sequential(\n")
				# counter to keep track of hidden_size attribute
				hdc = 0
				for i in range(2*int(configs['layers'])-1):
					if i % 2 == 0 and i == 0:
						file.write("\t\t\tnn." + configs['input_size' + (i+1) ] + "("+ configs['hidden_size'+hdc+1]  +"output_size)\n")
						hdc+=1
					elif i%2 ==0 and i == 2*int(configs['layers'])-1 :
						file.write("\t\t\tnn." + configs['hidden_size' + hdc] + "("+ configs['output_size']  +"output_size)\n")
					else:
						file.write("\t\t\tnn." + configs['hidden_size' + hdc ] + "("+ configs['hidden_size'+(hdc+1)]  +"hidden_size)\n")
						hdc +=1

				file.write("\t\t)")

				file.write("\tdef forward(self,x):\n")
				file.write("\t\tself.net(x)")

			self.fileName = "DQNet1"

	def getFileName(self):
		return self.fileName