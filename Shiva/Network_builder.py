import subprocess
import os

class NetworkBuilder ():
	def __init__ (self, input_size, output_size, configs):

		if configs["algorithm"] == "DQN":
			
			path = os.getcwd()
			# make a copy of dqnettemplate and call it fileName.py
			os.system("cp /home/phu/Documents/GitHub/Control-Tasks/Shiva/Network_Templates/DQNetTemplate.py " + path + "/Shiva/DQNet1.py")
			# /home/phu/Documents/GitHub/Control-Tasks/Shiva/Network_Templates/DQNetTemplate.py
			# use the configs to append the rest of the network to fileName.py
			with open(path + "/Shiva/DQNet1.py", "a") as file:
				file.write("\n        self.net = nn.Sequential(\n")
				# counter to keep track of hidden_size attribute
				hdc = 0
				spacing= "            "
				spacing2="        "
				for i in range(int(configs['layers'])):
					if i % 2 == 0 and i == 0:
						file.write(spacing + "nn." + str(configs['hidden_layer' +str(i+1)]) + "(input_size,"+ str(configs['hidden_size'+str(hdc+1)])  +"),\n")
						file.write(spacing + "nn." + str(configs['activation_function' + str(i+1)]) + "(),\n")
						hdc+=1
					elif i%2 ==0 and i == int(configs['layers'])-1 :
						file.write(spacing + "nn." + str(configs['hidden_layer' + str(hdc)]) + "("+ str(configs['hidden_size' + str(hdc)])+ ",output_size)")
					else:
						file.write(spacing + "nn." + str(configs['hidden_layer' + str(hdc)]) + "("+ str(configs['hidden_size'+ str(hdc)])+ ","+ str(configs['hidden_size'+ str(hdc+1)])  +"),\n")
						file.write(spacing + "nn." + str(configs['activation_function' + str(hdc)]) + "(),\n")
						hdc +=1

				file.write("\n" + spacing + ")")

				file.write("\n\n    def forward(self,x):\n")
				file.write(spacing2 + "return self.net(x)")


			self.fileName = "DQNet1"


	def getFileName(self):
		return self.fileName