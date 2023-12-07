def merge_sort(unsorted):
    if len(unsorted)>1:
        mid_point = int(len(unsorted)/2)
        first_half = unsorted[0:mid_point]
        second_half = unsorted[mid_point:]
        print('First half is:',first_half)
        print('Second half is:',second_half)
        merge_sort(first_half)
        merge_sort(second_half)
        a,b,c = 0,0,0
        while a<len(first_half) and b<len(second_half):
            if first_half[a]<second_half[b]:
                unsorted[c] = first_half[a]
                a+=1
                c+=1
            else:
                unsorted[c]=second_half[b]
                b+=1
                c+=1
        while a<len(first_half):
            unsorted[c]=first_half[a]
            a+=1
            c+=1
        while b<len(second_half):
            unsorted[c]=second_half[b]
            b+=1
            c+=1
        print('unsorted is :',unsorted)
        return unsorted
    else:
        return unsorted

aaa= [25,21,22,24,23,27,26,-1,0,100,1]

print(merge_sort(aaa))
