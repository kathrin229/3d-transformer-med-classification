import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = [[ 41,  65,  23],
 [  8, 138,  57],
 [ 15,  14,  88]]
df_cm = pd.DataFrame(array, index = ['CP', 'NCP', 'Normal'],
                  columns = ['CP', 'NCP', 'Normal'])
plt.figure(figsize = (7,5))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')