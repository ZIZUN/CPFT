import csv
with open('/home/leesm/Project/2022_2/contrastive_training/pretrain/data/HWU64/test_psuedo/label') as f:
    tr = csv.reader(f, delimiter='\t')
    # for row in tr:  
    #     print(row)
        
        
    with open('/home/leesm/Project/2022_2/contrastive_training/pretrain/data/HWU64/test_psuedo/label_', 'a') as file:
        temp = ''
        i = 0
        for row in tr:  
            
            if temp =='':
                file.write(str(i) + '\n')
                temp = row[0]
            elif temp !='' and temp == row[0]:
                file.write(str(i) + '\n')
            else:
                temp = row[0]
                i += 1
                file.write(str(i) + '\n')
                
                
            
            # print(row)
            # if temp == '1#':
            #     temp = row[0]
            #     file.write(str(i) + '\n')
            #     continue
            
            
            # if temp != row[0]:
            #     temp = row[0]
            #     i += 1
            #     file.write(str(i) + '\n')
            
            