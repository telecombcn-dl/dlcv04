import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

log = raw_input("Enter the path:")
log2 = raw_input("Enter the second path:")
loss=[]
val_loss=[]
acc=[]
val_acc=[]
loss2=[]
val_loss2=[]
acc2=[]
val_acc2=[]

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

for line in open(log2).readlines():
	aux_loss2 = 0
	aux_acc2 = 0
	nline = line.split(' - ')
	for l in nline:
		if l.startswith('loss: '):
			aux_loss2=l[6:11]
		elif l.startswith('acc: '):
			aux_acc2=l[5:10]
		elif l.startswith('val_loss: '):
			val_loss2.append(l[10:16])
			loss2.append(aux_loss2)
		elif l.startswith('val_acc: '):
			val_acc2.append(l[9:15])
			acc2.append(aux_acc2)

name = log2.split('.')

plt.figure(1)
x=range(1,len(loss)+1)
plt.plot(x,loss, marker="x", label="Training Loss Batch Normalization")
plt.plot(x,val_loss, marker="x", label="Validation Loss Batch Normalization")
plt.plot(x,loss2, marker="x", label="Training Loss No Batch Normalization")
plt.plot(x,val_loss2, marker="x", label="Validation Loss No Batch Normalization")
plt.legend()
plt.title("Batch Normalization vs No Batch Normalization")
plt.xlim([1.0,len(loss)])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig("plots/" + name[0] + "double_loss.png")



plt.figure(2)
x=range(1,len(acc)+1)
plt.plot(x,acc, marker="x", label="Training Accuracy  Batch Normalization")
plt.plot(x,val_acc, marker="x", label="Validation Accuracy Batch Normalization")
plt.plot(x,acc2, marker="x", label="Training Accuracy No Batch Normalization")
plt.plot(x,val_acc2, marker="x", label="Validation Accuracy No Batch Normalization")
plt.legend(loc="lower right")
plt.title("Batch Normalization vs No Batch Normalization")
plt.xlim([1.0,len(acc)])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("plots/" + name[0] + "double_accuracy.png")
