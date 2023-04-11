Huong dan cach chay code


cai dat thu vien va moi truong:


conda create --name biosyntax python==3.7.2
conda activate biosyntax
pip install -r requirements.txt


pip install benepar
python -m spacy download en_core_web_trf


python
>>> import benepar
>>> benepar.download('benepar_en3')



sau khi cai dat xong tat ca thu vien can thiet tu file requirements.txt cung nhu tai pre-train model Biobert (phien ban danh co tensorflow) vao folder biobert_v1.1_pubmed thi thuc hien cac buoc sau

B1: dau tien vao folder ConvertFile, copy data xml vafo trong folder data. (thay doi path trong 2 file python cho phu hop voi path data)
B2: file genPredicate.py dung de sinh ra cac file test theo tung predicate
B3: file genAll.py dung de sinh ra 3 file train dev test
B4: Hien trong code dang de chay predict theo tung predicate, de co the chay cho file test tong (file duoc sinh ra o buoc 3 thi sua lai vong lap for ) trong ham main(). copy all file data vao thu muc srl_data (thu muc duoc define trong file run.sh cua moi model)
		for file in listfile:
            tf.logging.info(f"*****  predicate {file} *****")
            name1 = file.split('.')[0]
            if "test_" not in name1:
                continue   
B5: trong moi model chay file sh cua no. hien trong file sh nay dang chay theo dang nohup, can chinh lai duong dan environment cua conda chinh xac de co the chay duoc model
B6: sau khi chay xong model thi chay file caculate_result2.py de tinh duoc diem P,R,F1
