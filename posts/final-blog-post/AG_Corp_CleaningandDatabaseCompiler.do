*Amanda Gregg, Middlebury College, 2019
*Main Data Cleaning and Compilation Code for "Factory Productivity and the Concession System of Incorporation in Late Imperial Russia"

clear all
set more off

*Set your working directory here
*cd [YOUR DIRECTORY HERE]

*
* 1894 Data Preparation
*

*Import the Data
import excel 1894MicroData.xlsx, sheet("Sheet1") firstrow

destring FoundingYear OntheSide Output OutputinItalics Women AdolescentWomen AdolescentMen MachineNumber1 MachinePower1 MachineNumber2 MachinePower2, replace force

*Generate a variable that codes whether the factory is owned by a corporation in 1900
gen Form1900 = 0
replace Form1900 = 1 if PSZ1900 ~= ""

*Separate European and Non-European Russia
gen EuropeanRussia = 0
replace EuropeanRussia = 1 if Province < 69

*Deal with Missing Values Appropriately: The following missing values are really zeros
replace Men = 0 if Men == .
replace Women = 0 if Women == . 
replace AdolescentMen = 0 if AdolescentMen == .
replace Boys = 0 if Boys == .
replace AdolescentWomen = 0 if AdolescentWomen == .
replace Girls = 0 if Girls == .
replace OntheSide = 0 if OntheSide == .
replace MachineNumber1 = 0 if MachineNumber1 == .
replace MachineNumber2 = 0 if MachineNumber2 == .
replace MachineNumber3 = 0 if MachineNumber3 == .
replace MachineNumber4 = 0 if MachineNumber4 == .
replace MachineNumber5 = 0 if MachineNumber5 == .
replace MachinePower1 = 0 if MachinePower1 == .
replace MachinePower2 = 0 if MachinePower2 == .
replace MachinePower3 = 0 if MachinePower3 == .
replace MachinePower4 = 0 if MachinePower4 == .
replace MachinePower5 = 0 if MachinePower5 == .

*Missing values is output are likely truly missing, but one could replicate the results
*with Output coded differently

*replace Output = 0 if Output == .
*replace OutputinItalics = 0 if OutputinItalics == .

*Similarly for total workers
*replace Total = 0 if Total == .

*Generate firm age

gen Age = 1895 - FoundingYear
gen LAge = log(Age)

*Aggregate some categories
egen Adolescents = rowtotal(AdolescentMen AdolescentWomen)
egen Children = rowtotal(Boys Girls)
gen HasLaborontheSide = 0
replace HasLaborontheSide = 1 if OntheSide ~= 0

gen TotalMachinePower = MachinePower1 +MachinePower2+ MachinePower3+ MachinePower4 +MachinePower5
gen TotalNumberofMachines = MachineNumber1 +MachineNumber2 +MachineNumber3 +MachineNumber4 +MachineNumber5

*Generate Output/Labor
gen OutputperWorker = Output / Total

*Generate proportions of workers and machines
gen PropAdultMen = Men / Total
gen PropAdultWomen = Women / Total
gen PropAdolMen = AdolescentMen/Total
gen PropAdolWomen = AdolescentWomen/Total
gen PropBoys = Boys / Total
gen PropGirls = Girls / Total
gen PropMale = (Men + AdolescentMen + Boy) / Total
gen PropFemale = (Women + AdolescentWomen + Girls) / Total
gen PropAdult = (Men + Women) / Total
gen PropAdol = Adolescents / Total
gen PropChildren = Children / Total

*Rename Regions and Industries so they are one word
replace Region = "CentralBlacksoil" if Region == "Central Blacksoil"
replace Region = "CentralIndustrial" if Region == "Central Industrial"
replace Industry = "Flax" if Industry == "Flax, Hemp, and Jute"
replace Industry = "MixedMaterials" if Industry == "Mixed Materials"
replace Industry = "MetalsandMachines" if Industry == "Metals and Machines"

*Generate categorical variables
	 
tab(Industry), gen(Ind)
tab(Region), gen(Reg)

foreach region in `regions' {
	gen `region' = 0
	replace `region' = 1 if Region == "`region'"
}

foreach industry in `industries' {
	gen `industry' = 0
	replace `industry' = 1 if Industry == "`industry'"
}

*Generate taxed activity variables
*Taxed activities: wine, yeasted wines, spirits, vodka and liquors, fruit wine and cognac, beer, mead, beet sugar, sugar refining, tobacco

local activities Wine Spirits Vodka Liquor Cognac FruitWine Beer Mead Sugar SugarRefining Tobacco

gen TaxedActivity = 0 

foreach activity in `activities' {
	gen Activity`activity' = 0
	forvalues number = 1/17 {
	replace Activity`activity' = 1 if Activity`number' == "`activity'"
	replace TaxedActivity = 1 if Activity`number' == "`activity'"
	}
}
	
save Cleaned1894MicroData.dta, replace

*
*
* 1900 Data Preparation
*
*

clear all
set more off

*First, import the data

import excel 1900MicroData.xlsx, firstrow

destring SubindustryCode YearFounded OrdersinParentheses WorkersinParentheses OutputinParentheses NumberofWorkers TotalYearlyOutput, replace force

*Similarly to the 1894 data, generate a variable to indicate whether a factory is owned by a corporation in 1908
gen Form1908 = 0
replace Form1908 = 1 if PSZ1908 != ""

replace Region = "Caucasus" if Region == "Caucausus"

*Similarly to above, one can replicate the results replacing missing values in output and workers with zeros
*replace NumberofWorkers = 0 if NumberofWorkers == .
*replace TotalYearlyOutput = 0 if TotalYearlyOutput == .

*Clean up certain uezds
replace CityorUezd = "Vilnius" if CityorUezd == "Vilna"
replace CityorUezd = "Vilnius" if CityorUezd == "Vilno"

gen YEAR = 1900

*Generate Firm Age
gen Age = 1900 - YearFounded

destring Province, replace force

*Save
save Cleaned1900MicroData.dta, replace

*
*
* 1908 Data Preparation
*
*

clear all

*Import the data file

import excel 1908MicroData.xlsx, sheet("Sheet1") firstrow

destring Province TotalYearlyOutput NumberofWorkers Orders Power, replace force

gen YEAR = 1908

*Special Note: Chernomorskaia changes regions between 1900 and 1908. In 1908, Chernomorskaia belongs to Southern.
gen str20 Region = ""
replace Region = "Southern" if Province == 68

*Region Names
replace Region = "Northern" if Province == 1
replace Region = "Northern" if Province == 63
replace Region = "Northern" if Province == 32
replace Region = "Northern" if Province == 33
replace Region = "Northern" if Province == 42

replace Region = "Eastern" if Province == 58
replace Region = "Eastern" if Province == 14
replace Region = "Eastern" if Province == 34
replace Region = "Eastern" if Province == 37
replace Region = "Eastern" if Province == 45
replace Region = "Eastern" if Province == 57

replace Region = "Prebaltic" if Province == 25
replace Region = "Prebaltic" if Province == 22 
replace Region = "Prebaltic" if Province == 50
replace Region = "Prebaltic" if Province == 10

replace Region = "Central Industrial" if Province == 61
replace Region = "Central Industrial" if Province == 13
replace Region = "Central Industrial" if Province == 19
replace Region = "Central Industrial" if Province == 30
replace Region = "Central Industrial" if Province == 31
replace Region = "Central Industrial" if Province == 49
replace Region = "Central Industrial" if Province == 56
replace Region = "Central Industrial" if Province == 66

replace Region = "Central Blacksoil" if Province == 64
replace Region = "Central Blacksoil" if Province == 23
replace Region = "Central Blacksoil" if Province == 35
replace Region = "Central Blacksoil" if Province == 36
replace Region = "Central Blacksoil" if Province == 44
replace Region = "Central Blacksoil" if Province == 46
replace Region = "Central Blacksoil" if Province == 48
replace Region = "Central Blacksoil" if Province == 52
replace Region = "Central Blacksoil" if Province == 55
replace Region = "Central Blacksoil" if Province == 5 
replace Region = "Central Blacksoil" if Province == 41
replace Region = "Central Blacksoil" if Province == 16

replace Region = "Northwestern" if Province == 59
replace Region = "Northwestern" if Province == 60
replace Region = "Northwestern" if Province == 11
replace Region = "Northwestern" if Province == 20
replace Region = "Northwestern" if Province == 28
replace Region = "Northwestern" if Province == 29

replace Region = "Southwestern" if Province == 62
replace Region = "Southwestern" if Province == 18
replace Region = "Southwestern" if Province == 40

replace Region = "Southern" if Province == 2
replace Region = "Southern" if Province == 4
replace Region = "Southern" if Province == 7
replace Region = "Southern" if Province == 6
replace Region = "Southern" if Province == 21
replace Region = "Southern" if Province == 17
replace Region = "Southern" if Province == 53

replace Region = "Previslitskii" if Province == 65
replace Region = "Previslitskii" if Province == 12
replace Region = "Previslitskii" if Province == 15
replace Region = "Previslitskii" if Province == 26
replace Region = "Previslitskii" if Province == 27
replace Region = "Previslitskii" if Province == 38
replace Region = "Previslitskii" if Province == 39 
replace Region = "Previslitskii" if Province == 43
replace Region = "Previslitskii" if Province == 47
replace Region = "Previslitskii" if Province == 51

replace Region = "Caucasus" if Province == 68
replace Region = "Caucasus" if Province == 3
replace Region = "Caucasus" if Province == 8
replace Region = "Caucasus" if Province == 67
replace Region = "Caucasus" if Province == 54
replace Region = "Caucasus" if Province == 24
replace Region = "Caucasus" if Province == 9

save Cleaned1908MicroData.dta, replace

*Append
use Cleaned1900MicroData.dta

append using Cleaned1908MicroData.dta

save Cleaned1900and1908MicroData.dta, replace


********************************************************************************
******************* DATABASE COMPILATION ***************************************
********************************************************************************


*Combine Data from 1894, 1900, and 1908 to make a panel
use Cleaned1894MicroData.dta

gen YEAR = 1894

append using Cleaned1900and1908MicroData.dta

*Merging to RUSCORP
*Bring forward previously matched PSZs
replace PSZ = PSZLastYear if PSZLastYear ~= ""

*Merge to RUSCORP (NOTE OF WARNING: If you want to use the firm's age in Ruscorp, remember to rename the YEAR variable in Ruscorp)
merge m:1 PSZ using AG_Corp_RuscorpMasterFile_Cleaned.dta, keepusing(STCAP STPRICE SHARES NEWDEV TYPE BONDS)
drop if _merge == 2
drop _merge

*gen SinceInc = YEAR - YEARRuscorp

*Keep variables we will use
keep newid YEAR PSZ PSZ1900 PSZ1908 FoundingYear ObsNoNew ObsNo1900New Province Region Output TotalYearlyOutput FormRuscorp PSZLastYear PSZ Total NumberofWorkers OntheSide TotalMachinePower Power TotalNumberofMachines Age TYPE Industry SubindustryCode STCAP TaxedActivity

*Drop provinces outside European Russia for consistency across three datasets
drop if Province > 68

*
*Reconcile Critical Variables
*Choices about missing values are important here
*

*Revenue
gen Revenue = .
replace Revenue = Output if YEAR == 1894
replace Revenue = TotalYearlyOutput if YEAR == 1900
replace Revenue = TotalYearlyOutput if YEAR == 1908
replace Revenue = . if Revenue == 0
drop Output TotalYearlyOutput

*Whether the Factory is Corporation-Owned
rename FormRuscorp Form
replace Form = 0 if Form == .
replace Form = 0 if Form == 2
replace Form = 1 if PSZLastYear ~= ""

*Total Workers
gen TotalWorkers = .
replace TotalWorkers = Total if YEAR ==1894
replace TotalWorkers = NumberofWorkers if YEAR == 1900
replace TotalWorkers = NumberofWorkers if YEAR == 1908
replace TotalWorkers = . if TotalWorkers == 0
drop Total NumberofWorkers

*Total Machine Power
gen TotalPower = .
replace TotalPower = TotalMachinePower if YEAR == 1894
replace TotalPower = Power if YEAR == 1908
drop TotalMachinePower Power
replace TotalPower = 0 if TotalPower == . & YEAR ~=1900
replace TotalPower = . if YEAR == 1900

*Total Number of Machines
gen TotalMachines = .
replace TotalMachines = TotalNumberofMachines if YEAR == 1894
replace TotalMachines = 0 if TotalMachines == . & YEAR == 1894
replace TotalMachines = . if YEAR ~=1894
drop TotalNumberofMachines TotalMachines

*Total Workers that Includes Outside Workers for 1894
gen GrandTotalWorkers = TotalWorkers
replace GrandTotalWorkers = TotalWorkers + OntheSide if YEAR == 1894

*Clean Up Age
replace Age = . if Age < 0

*Generate logs and other needed transformations

gen RevperWorker = Revenue / TotalWorkers
gen PowerperWorker = TotalPower / TotalWorkers

*Alternative definitions of workers
gen RevperGrandWorker = Revenue / GrandTotalWorkers
gen PowerperGrandWorker = TotalPower / GrandTotalWorkers

gen logRevperWorker = log(RevperWorker)
gen logPowerperWorker = log(PowerperWorker)

gen logRevperGrandWorker = log(RevperGrandWorker)
gen logPowerperGrandWorker = log(PowerperGrandWorker)

gen logRev = log(Revenue)
gen logWorkers = log(TotalWorkers)
gen logPower = log(TotalPower)

*Generate Region-Industry Groups and Province-Industry Groups

egen RegIndGroup = group(Region Industry)
egen RegIndYearGroup = group(Region Industry YEAR)
egen ProvIndGroup = group(Province Industry)
egen ProvIndYearGroup = group(Province Industry YEAR)
egen IndYearGroup = group(Industry YEAR)

egen IndustryFactor = group(Industry), label
egen ProvinceFactor = group(Province), label
egen YearFactor = group(YEAR), label

*Define corporation types
gen AKTS = .
replace AKTS = 0 if Form == 1
replace AKTS = 1 if TYPE == 1
replace AKTS = 1 if TYPE == 3
replace AKTS = 1 if TYPE == 5

gen PAI = .
replace PAI = 0 if Form == 1
replace PAI = 1 if TYPE == 2
replace PAI = 1 if TYPE == 4
replace PAI = 1 if TYPE == 6
drop TYPE

*Match factories across time using matching numbers
*Note that this paper only makes use of the primary match, ObsNoNew. The original
*data files include alternative matches as well, which some authors
*may consider using. 

generate factory_id = .
replace factory_id = ObsNoNew if YEAR == 1900
replace factory_id = ObsNo1900New if YEAR == 1894
replace factory_id = ObsNoNew if YEAR == 1908
replace factory_id = newid + 600000 if factory_id == .
drop ObsNoNew ObsNo1900New

*Identify factories that are going to incorporate

gen FormNextYear = 0
gen FormNextNextYear = 0

replace FormNextYear = 1 if PSZ1900 ~="" & YEAR == 1894
replace FormNextYear = 1 if PSZ1908 ~="" & YEAR == 1900
replace FormNextNextYear = 1 if PSZ1908 ~="" & YEAR == 1894

*The same factory in the same year should have FormNextYear coded the same way
bysort factory_id YEAR: egen FormNextYearb = max(FormNextYear)
bysort factory_id YEAR: replace FormNextYear = FormNextYearb
drop FormNextYearb

*Cleaning up Form and FormNextYear variables: factories that have FormNextYear = 1
*should be coded as corporations the next year. Factories that have Form = 1 in a subsquent
*year should be coded as switchers the previous year. Factories coded as corporations
*in any year should be coded as corporations in all *subsequent* years

foreach year in 1894 1900 1908 {
gen FactoryisCorpin`year'temp = 0
replace FactoryisCorpin`year'temp = 1 if YEAR == `year' & Form == 1
by factory_id: egen FactoryisCorpin`year' = max(FactoryisCorpin`year')
drop FactoryisCorpin`year'temp
gen FormNextYearin`year'temp = 0
replace FormNextYearin`year'temp = 1 if YEAR == `year' & FormNextYear == 1
by factory_id: egen FormNextYearin`year' = max(FormNextYearin`year')
drop FormNextYearin`year'temp
}

replace FormNextYear = 1 if Form == 0 & YEAR == 1894 & FactoryisCorpin1900 == 1
replace FormNextYear = 1 if Form == 0 & YEAR == 1900 & FactoryisCorpin1908 == 1

replace Form = 1 if YEAR == 1900 & FormNextYearin1894 == 1
replace Form = 1 if YEAR == 1908 & FormNextYearin1900 == 1

replace Form = 1 if YEAR == 1900 & FactoryisCorpin1894 == 1
replace Form = 1 if YEAR == 1908 & FactoryisCorpin1894 == 1
replace Form = 1 if YEAR == 1908 & FactoryisCorpin1900 == 1

*Also, give these corporations the appropriate PSZ Number and A-Corp or SP Type
*(Particularly a problem for the switchers)
*But only make the replacement if it's missing
by factory_id: egen AKTS_temp = max(AKTS)
replace AKTS = AKTS_temp if AKTS == .
drop AKTS_temp
replace AKTS = . if Form == 0

split PSZ1900, p("-")
split PSZ1908, p("-")
gen PSZ1900_temp = PSZ19001 + PSZ19002
gen PSZ1908_temp = PSZ19081 + PSZ19082
destring PSZ1900_temp PSZ1908_temp, replace force

by factory_id: egen switcherPSZ_a = max(PSZ1900_temp)
by factory_id: egen switcherPSZ_b = max(PSZ1908_temp)
replace switcherPSZ_a = 0 if switcherPSZ_a == .
replace switcherPSZ_b = 0 if switcherPSZ_b == .
gen switcherPSZ = switcherPSZ_a + switcherPSZ_b
tostring switcherPSZ, replace
replace switcherPSZ = "" if switcherPSZ == "0"
gen dig1 = substr(switcherPSZ,1,1)
gen dig2 = substr(switcherPSZ,2,6)
replace switcherPSZ = dig1 + "-" + dig2 if switcherPSZ~=""
replace switcherPSZ = "" if Form == 0
replace PSZ = switcherPSZ if Form == 1 & switcherPSZ~=""

merge m:1 PSZ using AG_Corp_RuscorpMasterFile_Cleaned.dta, keepusing(STCAP STPRICE SHARES NEWDEV TYPE BONDS)
drop if _merge == 2

replace AKTS = 0 if Form == 1 & switcherPSZ~=""
replace AKTS = 1 if TYPE == 1 & switcherPSZ~=""
replace AKTS = 1 if TYPE == 3 & switcherPSZ~=""
replace AKTS = 1 if TYPE == 5 & switcherPSZ~=""

drop _merge TYPE PSZ19001 PSZ19002 PSZ19081 PSZ19082 PSZ1900_temp PSZ1908_temp switcherPSZ_a switcherPSZ_b switcherPSZ dig1 dig2

*Reconcile Industry Labels and Generate Dummies
*General Industry Dummy Variables, using 1894 Variables as Base

gen Silk = 0
replace Silk = 1 if Industry == "Silk" & YEAR == 1900
replace Silk = 1 if Industry == "Silk" & YEAR == 1908

gen Flax = 0
replace Flax = 1 if Industry == "Flax" & YEAR == 1900
replace Flax = 1 if Industry == "Flax" & YEAR == 1908

gen Animal = 0
replace Animal = 1 if Industry == "Animal Products" & YEAR == 1900
replace Animal = 1 if Industry == "Animal Products" & YEAR == 1908
replace Industry = "Animal" if Industry == "Animal Products"

gen Wool = 0
replace Wool = 1 if Industry == "Wool" & YEAR == 1900
replace Wool = 1 if Industry == "Wool" & YEAR == 1908

gen Cotton = 0
replace Cotton = 1 if Industry == "Cotton" & YEAR == 1900
replace Cotton = 1 if Industry == "Cotton" & YEAR == 1908

gen MixedMaterials = 0
replace MixedMaterials = 1 if Industry == "Mixed Materials" & YEAR == 1900
replace MixedMaterials = 1 if Industry == "Mixed Materials" & YEAR == 1908
replace Industry = "Mixed Materials" if Industry == "MixedMaterials"

gen Wood = 0
replace Wood = 1 if Industry == "Wood" & YEAR == 1900
replace Wood = 1 if Industry == "Wood" & YEAR == 1908

gen Paper = 0
replace Paper = 1 if Industry == "Paper" & YEAR == 1900
replace Paper = 1 if Industry == "Paper" & YEAR == 1908

gen MetalsandMachines = 0
replace MetalsandMachines = 1 if Industry == "Metals and Machines" & YEAR == 1900
replace MetalsandMachines = 1 if Industry == "Metals" & YEAR == 1908
replace Industry = "Metals and Machines" if Industry == "MetalsandMachines"
replace Industry = "Metals and Machines" if Industry == "Metals"

gen Foods = 0
replace Foods = 1 if Industry == "Foods A" & YEAR == 1900
replace Foods = 1 if Industry == "Foods A" & YEAR == 1908
replace Industry = "Foods A" if Industry == "Foods"

gen Chemical = 0
replace Chemical = 1 if Industry == "Chemicals" & YEAR == 1900
replace Chemical = 1 if Industry == "Chemicals" & YEAR == 1908
replace Industry = "Chemicals" if Industry == "Chemical"

gen Mineral = 0
replace Mineral = 1 if Industry == "Mineral Products" & YEAR == 1900
replace Mineral = 1 if Industry == "Mineral Products" & YEAR == 1908
replace Industry = "Mineral Products" if Industry == "Mineral"

drop if Industry == "Electrical Stations"
drop if Industry == "Foods B"
drop if Industry == "Mining"

*Reconcile Region Names
replace Region = "CentralBlacksoil" if Region == "Central Blacksoil"
replace Region = "CentralIndustrial" if Region == "Central Industrial"

*Drop extraneous variables
drop FormNextYearin1908

*
*Generate variable labels
*
rename newid id 
label var id "Observation id number, updated in 2018"
label var Form "Factory is a corporation if Form == 1. (Matched to Ruscorp)"
label var PSZ "Corporation's number in the Polnoe Sobranie Zakonov"
label var PSZ1900 "1894 factory's PSZ number as a corporation in 1900"
label var PSZ1908 "1900 factory's PSZ number as a corporation in 1908"
label var FoundingYear "Factory founding year, available for 1894 and 1900"
label var Province "Factory's province number 1 through 68"
label var Region "Factory's region"
label var Industry "Factory's industry category"
label var SubindustryCode "Factory's more finely defined industry; only for 1900 and only for Online Appendix analyses"
label var OntheSide "Number of workers the factory employs outside the factory in 1894"
label var Age "Factory's age"
label var TaxedActivity "Factory's activity is likely subject to the excise (aktsiz) tax"
label var YEAR "Factory's year in the data"
label var PSZLastYear "Corporation-owned factory's PSZ number in the last year of the data"
label var Revenue "Factory's total revenue"
label var TotalWorkers "Factory's total workers"
label var TotalPower "Factory's total machine horsepower; available in 1894 and 1908"
label var GrandTotalWorkers "Factory's total workers plus 1894 workers outside factory"
label var RevperWorker "Factory's revenue per worker"
label var PowerperWorker "Factory's total machine horsepower per worker"
label var RevperGrandWorker "Factory's revenue per worker, including workers outside the factory"
label var PowerperGrandWorker "Factory's total machine horsepower per worker, including workers outside the factory"
label var logRevperWorker "Factory's log (revenue / workers), including workers outside the factory"
label var logPowerperWorker "Factory's log (power / workers), including workers outside the factory"
label var logRevperGrandWorker "Factory's log (revenue / workers)"
label var logPowerperGrandWorker "Factory's log (power / workers)"
label var logRev "Factory's log Revenue"
label var logWorkers "Factory's log Workers"
label var logPower "Factory's log Machine Horsepower"
label var AKTS "Corporation calls its shares Aktsiia"
label var PAI "Corporation calls its shares Pai"
label var factory_id "Establishment id, for matching factories over time"
label var FormNextYear "Factory is a corporation in the next year of data"
label var FormNextNextYear "1894 Factory is a corporation in 1908"
label var FactoryisCorpin1894 "Factory is a corporation in 1894"
label var FactoryisCorpin1900 "Factory is a corporation in 1900"
label var FactoryisCorpin1908 "Factory is a corporation in 1908"
label var FormNextYearin1894 "In 1894, the factory would be a corporation in 1900"
label var FormNextYearin1900 "In 1900, the factory would be a corporation in 1908"
label var Silk "=1 if factory is in the Silk industry"
label var Flax "=1 if factory is in the Flax industry"
label var Animal "=1 if factory is in the Animal Products industry"
label var Wool "=1 if factory is in the Wool industry"
label var Cotton "=1 if factory is in the Cotton industry"
label var MixedMaterials "=1 if factory is in the Mixed Materials industry"
label var Wood "=1 if factory is in the Wood industry"
label var Paper "=1 if factory is in the Paper industry"
label var MetalsandMachines "=1 if factory is in the Metals and Machines industry"
label var Foods "=1 if factory is in the Foods industry"
label var Chemical "=1 if factory is in the Chemicals industry"
label var Mineral "=1 if factory is in the Mineral Products industry"

*Save file with taxed activities for appendix work
save AG_Corp_Prod_Database_withAktsiz.dta, replace

*Drop taxed activities in the 1894 Foods A Industry, as requested by a referee in Revision 1 
*Note that we have already dropped taxed activities in other years by dropping the "Foods B" industry above.
drop if TaxedActivity == 1 & Industry == "Foods A"

save AG_Corp_Prod_Database.dta, replace
