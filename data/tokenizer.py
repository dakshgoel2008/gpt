from typing import List
from collections import defaultdict

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        chars = [c for c in corpus]
        ans=[]
        for _ in range(num_merges):
            if(len(chars)<2):
                break

            mp=defaultdict(int)
            for i in range(len(chars) - 1):
                pair=(chars[i],chars[i+1])
                mp[pair]=mp.get(pair,0)+1
            
            maxi=max(mp.values())
            pairs=sorted(p for p,c in mp.items() if c==maxi)
            best=pairs[0]

            ans.append([best[0],best[1]])

            new_token=[]
            i = 0
            while i<len(chars):
                if i<len(chars)-1 and chars[i]==best[0] and chars[i+1]==best[1]:
                    new_token.append(best[0]+best[1])
                    i+=2
                else:
                    new_token.append(chars[i])
                    i+=1
            chars=new_token
            
        return ans
