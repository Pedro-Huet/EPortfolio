use "C:\Users\USER\Documents\Diplomado Econometría\Módulo 4 Econometría\Trabajo Final\Base de Datos Brazil\Brazil Education Panel Database\Panel Data\Brazil Education Panel Database\BrazilEduPanel_School.dta"

*Dependiente:

drop if RATE_ABANDON ==.

recode RATE_ABANDON (0/4.999= 0) (5/100= 1), gen(abandono_esc_alto)

*Independientes:

drop if PROFFUNDTOT ==.

recode PROFFUNDTOT (101/262 = 101 "Más de 100"), gen(num_profes)

drop if EDUCTEACH ==.

recode EDUCTEACH (8/8.4999= 8 "8 años") (8.5/9.4999 = 9 "9 años") (9.5/10.4999 = 10 "10 años") (10.5/11.4999 = 11 "11 años") (11.5/12.4999 = 12 "12 años") (12.5/13.4999 = 13 "13 años") (13.5/14.4999 = 14 "14 años") (14.5/15 = 15 "15 años o más"), gen(escolaridad_prof)

drop if CLASSSIZE ==.

recode CLASSSIZE (1/1.4999= 1) (1.5/2.4999= 2) (2.5/3.4999= 3) (3.5/4.4999= 4) (4.5/5.4999= 5) (5.5/6.4999= 6) (6.5/7.4999= 7) (7.5/8.4999= 8) (8.5/9.4999= 9) (9.5/10.4999= 10) (10.5/11.4999= 11) (11.5/12.4999= 12) (12.5/13.4999= 13) (13.5/14.4999= 14) (14.5/15.4999= 15) (15.5/16.4999= 16) (16.5/17.4999= 17) (17.5/18.4999= 18) (18.5/19.4999= 1) (19.5/20.4999= 20) (20.5/21.4999= 21) (21.5/22.4999= 22) (22.5/23.4999= 23) (23.5/24.4999= 24) (24.5/25.4999= 25) (25.5/26.4999= 26) (26.5/27.4999= 27) (27.5/28.4999= 28) (28.5/29.4999= 29) (29.5/30.4999= 30) (30.5/31.4999= 31) (31.5/32.4999= 32) (32.5/33.4999= 33) (33.5/34.4999= 34) (34.5/35.4999= 35) (35.5/36.4999= 36) (36.5/37.4999= 37) (37.5/38.4999= 38) (38.5/39.4999= 39) (39.5/40.4999= 40) (40.5/41.4999= 41) (41.5/42.4999= 42) (42.5/43.4999= 43) (43.5/44.4999= 44) (44.5/45.4999= 45) (45.5/46.4999= 46) (46.5/47.4999= 47) (47.5/48.4999= 48) (48.5/49.4999= 49) (49.5/50.4999= 50) (50.5/500= 51 "más de 50"), gen(alumnos_por_salon)    

drop if NU_COMPUTADOR ==.

recode NU_COMPUTADOR (201/5600 = 201 "Más de 200"), gen(num_computadoras)

*Temporal:

drop if NU_AN < 2003

sort CO_ENTIDADE

by CO_ENTIDADE: gen count=_N

tab count

drop if count < 12

rename NU_ANO Year

rename CO_ENTIDADE cod_escuela

xtset cod_escuela Year

xtsum abandono_esc_alto num_profes escolaridad_prof alumnos_por_salon num_computadoras

xtline abandono_esc_alto

xtline num_pro

xtline num_pro num_compu escolaridad alumnos

xtline num_pro num_compu

xtline escolaridad

xtline alumnos

