## yapf
bash path_to_project/check_format.sh

## isort
### check
isort -rc -w -d 80 ./
### apply
isort -rc -w 80 ./

## flake
### check
autoflake -r ./ -i
### apply
autoflake -r ./
