import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

log = raw_input("Enter the path:")
loss=[]
val_loss=[]
acc=[]
val_acc=[]
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

plt.figure(1)
x=range(1,len(loss)+1)
plt.plot(x,loss, marker="x", label="Training Loss")
plt.plot(x,val_loss, marker="x", label="Validation Loss")
plt.legend()
plt.xlim([1.0,len(loss)])
# plt.title('Batch 32')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig("plots/" + name[0] + "_loss.png")

plt.figure(2)
x=range(1,len(acc)+1)
plt.plot(x,acc, marker="x", label="Training Accuracy")
plt.plot(x,val_acc, marker="x", label="Validation Accuracy")
plt.legend(loc="lower right")
plt.xlim([1.0,len(acc)])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("plots/" + name[0] + "_accuracy.png")
