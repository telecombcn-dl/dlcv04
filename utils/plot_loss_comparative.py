import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

original = "original.txt"

log = raw_input("Enter the path:")

original_loss=[]
original_val_loss=[]
original_acc=[]
original_val_acc=[]

for line in open(log).readlines():
	original_aux_loss = 0
	original_aux_acc = 0
	nline = line.split(' - ')
	for l in nline:
		if l.startswith('loss: '):
			original_aux_loss=l[6:11]
		elif l.startswith('acc: '):
			original_aux_acc=l[5:10]
		elif l.startswith('val_loss: '):
			original_val_loss.append(l[10:16])
			original_loss.append(original_aux_loss)
		elif l.startswith('val_acc: '):
			original_val_acc.append(l[9:15])
			original_acc.append(original_aux_acc)

name = log.split('.')

loss=[]
val_loss=[]
acc=[]
val_acc=[]

for line in open(original).readlines():
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
plt.plot(x,original_loss, marker="x", label="Training Loss without modifications")
plt.plot(x,original_val_loss, marker="x", label="Validation Loss without modifications")
plt.legend()
plt.xlim([1.0,len(loss)])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig("plots/" + name[0] + "_loss.png")

plt.figure(2)
x=range(1,len(acc)+1)
plt.plot(x,acc, marker="x", label="Training Accuracy")
plt.plot(x,val_acc, marker="x", label="Validation Accuracy")
plt.plot(x,original_acc, marker="x", label="Training Accuracy without modifications")
plt.plot(x,original_val_acc, marker="x", label="Validation Accuracy without modifications")
plt.legend(loc="lower right")
plt.xlim([1.0,len(acc)])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("plots/" + name[0] + "_accuracy.png")
