*I.- Baseline Balance Check

import delimited "C:\Users\pedro\Videos\Work\Application\Balanced Baselines\r_schools.csv", clear 

logit treatment district location n_teachers n_teachers_fem female_head_teacher n_students_fem n_students_male n_schools_2km av_teacher_age av_student_score n_latrines
margins, dydx(*)

*In general, balanced baseline, but number of latrines could be suspicious.


*II.- Data Analysis

*i) Short-term impact of School Subsidies 

import delimited "C:\Users\pedro\Videos\Work\Application\Analysis\r_student_baseline_year_merged.csv", clear  

*a) Primary performance metrics

logit dropout trt_school yob sex i.visit_month if post == 0

margins, dydx(*)

	
*The program does seem to have a positive and statistically significant effect in reducing school evasion, yet it seems relatively small compared to other factors.


*b) Secondary performance metrics

logit pregnt trt_school yob sex i.visit_month if post == 0
margins, dydx(*)

logit married trt_school yob sex i.visit_month if post == 0
margins, dydx(*)

*As with school evasion, it would also seem that the subsidies program has a positive and statistically significant effect in school teen pregnancy and marriage, yet it also seems relatively small.


*c) Possible Interaction Effects of the Subsidy (Mentioned, but placed in Appendix)

logit dropout sex##trt_school yob i.visit_month if post == 0

logit dropout yob##trt_school sex i.visit_month if post == 0

*It would seem that some interactions between the main variables don't offer additional insights.


*ii) Long-term impact of School Subsidies 

*a) Primary performance metrics

logit dropout trt_school yob sex i.visit_month if post == 1

margins, dydx(*)

*The program does seem to have a positive and statistically significant effect in reducing school evasion, yet it seems relatively small compared to other factors.


*b) Secondary performance metrics

logit pregnt trt_school yob sex i.visit_month if post == 1
margins, dydx(*)

logit married trt_school yob sex i.visit_month if post == 1
margins, dydx(*)

*As with school evasion, it would also seem that the subsidies program has a positive and statistically significant effect in reducing teen pregnancy and marriage,, yet it also seems relatively small.


*c) Possible Interaction Effects of the Subsidy (Mentioned, but placed in Appendix)

logit dropout sex##trt_school yob i.visit_month if post == 1

logit dropout yob##trt_school sex i.visit_month if post == 1

*It would seem that some interactions between the main variables don't offer additional insights.


*iii) Difference-in-Difference Approach

reg dropout trt_school##post, robust


*iv) Relationship between school marriage, children and dropouts?

logit dropout married children trt_school if post == 0

logit dropout married children trt_school if post == 1
margins, dydx(*)
