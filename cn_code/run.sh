export PROJECTHOME=/kd-future/kdfast-strategy
export kdfast_config=$PROJECTHOME/conf/database.yaml
export kdconfig=$kdfast_config
export kdfast_ini=$PROJECTHOME/conf/kdfast_conf.ini
export PYTHONPATH=$PYTHONPATH:$PROJECTHOME

source /venv/venv-kdfast/bin/activate

#python /cta-model-template/zhaoyu_china/原始中国市场实证.py
#python /cta-model-template/zhaoyu_china/PCA中国市场实证.py
#python /cta-model-template/zhaoyu_china/PLS中国市场实证.py
python /cta-model-template/zhaoyu_china/AE中国市场实证.py
