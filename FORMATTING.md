## yapf
bash path_to_project/check_format.sh
### check
yapf -p -r -d --style='{COLUMN_LIMIT:80}' ./
### apply
yapf -p -r -i --style='{COLUMN_LIMIT:80}' ./

## isort
### check
isort -rc -w 80 -d ./
### apply
isort -rc -w 80 ./

## flake
### check
autoflake -r ./
### apply
autoflake -r ./ -i
