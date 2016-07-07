import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

log = raw_input("Enter the path:")
log1 = raw_input("Enter the path:")
loss=[]
val_loss=[]
acc=[]
val_acc=[]
loss1=[]
val_loss1=[]
acc1=[]
val_acc1=[]


for line in open(log).readlines():
	aux_loss = 0
	aux_acc = 0
	nline = line.split(' - ')
	for l in nline:
		if l.startswith('loss: '):
			aux_loss=l[6:11]
		elif l.startswith('acc: '):
			aux_acc=l[5:10]
		elif l.startswith('val_loss: '):
			val_loss.append(l[10:16])
			loss.append(aux_loss)
		elif l.startswith('val_acc: '):
			val_acc.append(l[9:15])
			acc.append(aux_acc)

name = log.split('.')

for line in open(log1).readlines():
	aux_loss = 0
	aux_acc = 0
	nline = line.split(' - ')
	for l in nline:
		if l.startswith('loss: '):
			aux_loss=l[6:11]
		elif l.startswith('acc: '):
			aux_acc=l[5:10]
		elif l.startswith('val_loss: '):
			val_loss1.append(l[10:16])
			loss1.append(aux_loss)
		elif l.startswith('val_acc: '):
			val_acc1.append(l[9:15])
			acc1.append(aux_acc)

name = log.split('.')

"""plt.figure(1)
x=range(1,len(loss)+1)
plt.plot(x,loss, marker="x", label="Training Loss")
plt.plot(x,val_loss, marker="x", label="Validation Loss")
plt.legend()
plt.xlim([1.0,len(loss)])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig("plots/" + name[0] + "_loss.png")
"""
plt.figure(2)
x=range(1,len(acc)+1)
plt.plot(x,acc, marker="x", label="Training Accuracy_32")
plt.plot(x,val_acc, marker="x", label="Validation Accuracy_32")
plt.plot(x,acc1, marker="x", label="Training Accuracy_128")
plt.plot(x,val_acc1, marker="x", label="Validation Accuracy_128")
plt.legend(loc="lower right")
plt.xlim([1.0,len(acc)])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("plots/" + name[0] + "_accuracy.png")
