#/bash
echo "这是一个统计提交行数的脚本"
read -p "提交代码的邮箱：" email
read -p "统计的起始时间（不填默认2021-01-01）:" begin
if [ -z $email ]
then
	email="1695949332@qq.com"
fi
if [ -z $begin ]
then
	begin="2021-01-01"
fi
git log --author=$email --since=$begin --pretty=tformat: --numstat | gawk '{add += $1 ; subs += $2 ; loc += $1 - $2} END {printf "增加的行数 ：%s 删除的行数：%s 总行数：%s\n",add,subs,loc}'
