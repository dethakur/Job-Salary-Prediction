import pickle
data_val = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/k-means-data-complete.pkl','rb'))

newData = {}

old_val = data_val
# del old_val['FullDescription']
# del data_val['FullDescription']

print("map constructing")  

for cluster_id in old_val.keys():
    el = old_val[cluster_id]
    del el['FullDescription']
    
    mapset = {}
    for clust_id in old_val.keys():
        if clust_id == cluster_id:
            continue
        
        for key_val in old_val[clust_id].keys():
            if key_val not in mapset:
                mapset[key_val] = []
            
            arr = []
            if key_val == 'Title' or key_val == 'Category' or key_val == 'Company' and False:
                for el in old_val[clust_id][key_val]:
                    arr = arr + el.split() 

                mapset[key_val] = list(set(mapset[key_val]).union(set(arr)))
            else:
                mapset[key_val] = list(set(mapset[key_val]).union(set(old_val[clust_id][key_val])))
            
    print("map constructed")  
    
    for key in old_val[cluster_id].keys():
        print('cluster id = {} , key = {} , original = {}'.format(cluster_id,key,len(data_val[cluster_id][key])))
        if key_val == 'Title' or key_val == 'Category'  or key_val == 'Company' and False:
            arr = []
            for el in old_val[cluster_id][key]:
                arr = arr + el.split()
            data_val[cluster_id][key] = list(set(arr) - set(mapset[key_val]))    
        else:
            data_val[cluster_id][key] = list(set(data_val[cluster_id][key]) - set(mapset[key_val]))
        print('cluster id = {} , key = {} ,new = {}'.format(cluster_id,key,len(data_val[cluster_id][key])))


fileArr = {}
for i in data_val.keys():    
    for key in data_val[i].keys():
        if key == 'FullDescription':
            continue


        f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/info-files/cluster-{}/{}.txt'.format(i,key),'w')
        for el in data_val[i][key]:
            f.write(str(el)+"\n")