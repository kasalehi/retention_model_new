-- according to my invistigation we are going to create the first dataset source as active and inactive status which they
-- are take huge portion of our members . apart from that, make the data source conssitant helps model to gives us the better
-- prediction. therefore we are trying to fouces on those memebrs (respectively subscriptions) to check
-- in order to avoid abnormality we use status reason inactive and active which pulling yearly about 17000 leaves and rest stay tunned.
-- if we bring other features makes the data abnormal so that is main reason why we avoid that one. 

-- below is query which retruns those normal data and might we need add more features which i think at this stage we dont need 


use LesMills_Reporting
go

with total as (
select cm.LMID,
cm.MembershipCategory,
cm.MembershipOrigin,
cm.MembershipStatusReason,
cm.MembershipSubCategory,
cm.MembershipTerm,
cm.MembershipType,
cm.PaymentFrequency,
cm.RegularPayment,
ci.Gender,
Round(DATEDIFF(year, ci.DateOfBirth, GETDATE()),0) as Age,
sum(att.WeekVisits) as TotalAttendance,
case 
    when (MembershipStatusReason in ('Inactive') and MembershipOrigin='New' and (MembershipEndDate between '2024-01-01' and '2025-01-01'))
	then '1'
	when (MembershipStatusReason='Active' and (MembershipEndDate is null or MembershipEndDate>='2026-01-01') and MembershipStartDate<='2024-01-01') then '0'
	else 'NotIncluded'
end as 'Churned'
from repo.CRM_ActiveMemberships cm
left join repo.CRM_MemberContact_Info ci
on cm.LMID=ci.LMID
join pbi.LMNZ_MemberWeeklyAttendanceCount att
on cm.MembershipID=att.MembershipID
where cm.MembershipTerm in ('12 Months', '12 Month', '24 Month', '24 Months')
group by 
cm.LMID,
cm.MembershipCategory,
cm.MembershipOrigin,
cm.MembershipStatusReason,
cm.MembershipSubCategory,
cm.MembershipTerm,
cm.MembershipType,
cm.PaymentFrequency,
cm.RegularPayment,
ci.Gender,
Round(DATEDIFF(year, ci.DateOfBirth, GETDATE()),0),
case 
    when (MembershipStatusReason in ('Inactive') and MembershipOrigin='New' and (MembershipEndDate between '2024-01-01' and '2025-01-01'))
	then '1'
	when (MembershipStatusReason='Active' and (MembershipEndDate is null or MembershipEndDate>='2026-01-01') and MembershipStartDate<='2024-01-01') then '0'
	else 'NotIncluded'
end
)
select  *  from total where Churned!='NotIncluded'



















