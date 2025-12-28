for i in range(1,7):
    print(i)
####
i=2
while i<=12:
    print(i)
    i=i+2
#####
i=1
while i<=6561:
    print(i)
    i=i*3
####
i="a"
for k in range(6):
    print(i)
    i=i+"a"

a="5"
for x in range(1,10):
    print(a)
    a=a+str(x)
####
def cevre_hesapla(r,pi=3):
    if pi<3.2 and pi>3:
        print("çevre:",2*pi*r)
    else:
        print("lütfen pi sayısını 3 ile 3,2 arasında giriniz:")
####
def dosya_ac(dosya_adı):
    try:
        f=open(dosya_adı,"r")
    except:
        f=open(dosya_adı,"x")
####

def cevre_hesapla(r,pi=3):
    if pi<3.2 and pi>=3:
        print("çevre:",2*pi*r)
    else:
        print("lütfen pi sayısını 3 ile 3,2 arasında giriniz:")
sayı=int(input("lütfen sayı giriniz"))

cevre_hesapla(sayı)