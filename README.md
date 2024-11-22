# Lipreading using Temporal Convolutional Networks - Working Fork

Перед инференсом используйте:

```Shell
pip install -r requirements.txt
```

Для инференса необходимо скачать стейт модели по ссылке - https://drive.google.com/file/d/1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm/view

В качестве конфига модели необходимо указать `configs/lrw_resnet18_dctcn_boundary.json`

Чтобы получить эмбеддинги необходимо запустить следующую команду, в `<MOUTH-PATCH-PATH>` должна быть директория `mouths`, обработанные эмбеддинги будут сохранены в `<MOUTH-PATCH-PATH>\mouths_embeds`

```Shell
python3 main.py --modality video \
                                      --extract-feats \
                                      --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --mouth-patch-path <MOUTH-PATCH-PATH>
```
