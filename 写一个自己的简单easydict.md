# 写一个自己的简单easydict

最近看代码的时候发现一个简单有趣的 python 包 easydict，这个包可以让我们像操作对象的属性一样来操作 dict 中的键值。  
感觉还是挺实用的，简单的看了下 easydict 的代码，发现 easydict 的实现还是稍微有点耗内存，同样的键值关系保存了两次，  
就想自己动手写一个简单的 helloworld 版 easydict。

## 代码示例

~~~python
class MyEasyDict(dict):

    def __init__(self, odict = None):
        super(MyEasyDict, self).__init__()
        if odict and not self._is_only_dict(odict):
            raise Exception("the param must be a dict")
        if odict:
            for k, v in odict.items():
                if isinstance(v, list) or isinstance(v, tuple):
                    self[k] = [MyEasyDict(e) if self._is_only_dict(e) else e for e in v]
                elif self._is_only_dict(v):
                    self[k] = MyEasyDict(v)
                else:
                    self[k] = v

    def _is_only_dict(self, v):
        return isinstance(v, dict) and not isinstance(v, self.__class__)

    def __setattr__(self, key, value):
        if self._is_only_dict(value):
            self[key] = MyEasyDict(value)
        else:
            self[key] = value

    def __getattr__(self, item):
        if item not in self.keys():
            raise Exception("not contain this key")
        return self[item]

    def __delattr__(self, item):
        if item not in self.keys():
            raise Exception("key not in dict")
        del self[item]

if __name__ == "__main__":
    my_dict = MyEasyDict()
    my_dict.a = {"b" : 10}
    print(my_dict.a.b)
    my_dict = MyEasyDict({'a' : {'b' : 100}})
    print(my_dict.a.b)
    del my_dict.a.b
    #print(my_dict.a.b)

~~~
