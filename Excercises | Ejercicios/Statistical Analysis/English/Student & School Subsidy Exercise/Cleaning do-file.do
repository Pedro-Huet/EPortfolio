import delimited "C:\Users\pedro\Videos\Work\Application\Balanced Baselines\schools.csv"

replace location = 0 if location == 1
replace location = 1 if location == 2

export delimited using "C:\Users\pedro\Videos\Work\Application\Balanced Baselines\r_schools.csv", replace


import delimited "C:\Users\pedro\Videos\Work\Application\Analysis\student_baseline.csv", clear 

replace yob = . if yob == 9999

export delimited using "C:\Users\pedro\Videos\Work\Application\Analysis\r_student_baseline.csv", replace


import delimited "C:\Users\pedro\Videos\Work\Application\Analysis\student_followups.csv", clear 

replace died = "" if died == "-99"
replace died = "" if died == "NA"

replace pregnant = "" if pregnant == "-99"
replace pregnant = "" if pregnant == "NA"

replace married = "" if married == "-99"
replace married = "" if married == "NA"

replace children = "" if children == "-99"
replace children = "" if children == "NA"

replace dropout = "" if dropout == "-99"
replace dropout = "" if dropout == "NA"

export delimited using "C:\Users\pedro\Videos\Work\Application\Analysis\r_student_baseline_year_merged.csv", replace