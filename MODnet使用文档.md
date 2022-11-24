# MODnet使用文档

1. 下载源代码

```
git clone https://github.com/ZHKKKe/MODNet
cd MODNet
```

2. 将下载好的预训练模型放在```pretrained```目录下
3. 将```inference.py```放在代码根目录下
4. 运行inference代码：
```shell
python inference.py \
--input-path \path\to\input\image\dir \
--out-path \path\to\output\image\dir \
--ckpy-path \path\to\model\checkpoint
```