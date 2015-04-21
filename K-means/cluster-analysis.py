import pickle
import copy
data_val = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/k-means-data-complete.pkl','rb'))

newData = {}

old_val = copy.deepcopy(data_val)
# del old_val['FullDescription']
# del data_val['FullDescription']
for i in data_val.keys():    
    print("Cluster {} writing".format(i))
    for key in data_val[i].keys():
        # if key == 'FullDescription':
        #     continue
        f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/info-files/cluster-{}/{}.txt'.format(i,key),'w')
        l = []
        for el in data_val[i][key]:
            l.append(str(el))
            # f.write(str(el)+"\n")

    print("Cluster {} written".format(i))