import keras_ocr


img_path1 = "./data/act_time_country.png"

pipeline = keras_ocr.pipeline.Pipeline()
extract_info = pipeline.recognize([img_path1])
print(extract_info[0][0])



diz_cols = {'word':[],'box':[]}
for el in extract_info[0]:
    diz_cols['word'].append(el[0])
    diz_cols['box'].append(el[1])
kerasocr_res = pd.DataFrame.from_dict(diz_cols)
kerasocr_res