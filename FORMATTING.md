## yapf
bash path_to_project/check_format.sh

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
