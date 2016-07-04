mkdir -p ../datasets
mkdir -p ../datasets/terrassa
wget  https://imatge.upc.edu/web/sites/default/files/projects/1634/public/terrassa-buildings/terrassa900-trainval.zip -O ../datasets/terrassa/terrassa900-trainval.zip
wget  https://imatge.upc.edu/web/sites/default/files/projects/1634/public/terrassa-buildings/terrassa900-test.zip -O ../datasets/terrassa/terrassa900-test.zip
unzip terrassa900-test.zip -d test
unzip terrassa900-train.zip -d train