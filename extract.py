import requests
from bs4 import BeautifulSoup

URL = 'http://www.gulf-times.com/story/524687/India-s-demonetisation-drive-drags-down-Nepal-s-ec&sa=U&ved=0ahUKEwj_3tej7dvSAhUMr48KHVweDeYQpwIIIzAF&usg=AFQjCNFdKBbE-4L9JxmMddqjsO0TeDRotw'


def runn(**params):
    response = requests.get(URL.format(**params))
    str =  response.content
    print len(str)
    # print len(response.content)
    fo = open("foo1.html", "w")
    fo.write( str )
    # print "hello"
    fo.close()
    soup = BeautifulSoup(str, 'html.parser')
    mysoup = soup
    # for link in soup.find_all('a'):
    # 	print(link.get('href'))

    mydivs = mysoup.find_all("p",{"style" : "font-size : 15px!important"})
    str1 = mydivs[0]
    str2 = str1.get_text(' ', strip=True)
    print(type(str2))

    import io
    encoding = 'utf-8'
    with io.open("content.txt", 'w', encoding=encoding) as f:
        f.write(str2)
    
    # for num in range(1,len(str1)):
    #  	strings += str(str1[num])
    # for i in str1:
    # 	strings += s

    # fo1 = open("foo1.txt", "w")
    # fo1.write( str )
    # # print "hello"
    # fo1.close()
    #<td>My home address</td>
	# td.contents



runn()
