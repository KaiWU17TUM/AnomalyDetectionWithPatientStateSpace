library(data.table)
library(ggplot2)
library(tidyr)
library(dplyr)
library(ggpubr)
library(GGally)

setwd('/home/kai/DigitalICU/HIRID-PatientStateSpace/processed/r_plotting')
source('../../src/config.R')

dt <- fread('pharma_physio_correlation1h.csv', drop='V1')
dt <- as.data.table(dt)

dt_physio <- copy(dt[, 10:69])
dt_physio[dt_physio==-1] = NA
 
theme_rotatex = theme_bw() + theme(axis.text.x = element_text(angle = 30, vjust = 1, hjust=1)) + theme(axis.text = element_text(size = 12), axis.title = element_text(size = 20))    

# correlation pharma - patient information
dt_info <- distinct(copy(dt[, c(..COL_INFO)]))
dt_info <- dt_info[, pharmaid:=as.factor(pharmaid)] 

dt_pharma_apache <- dt_info[, .(apache, pharmaid, dischargestatus)]

options(repr.plot.width = 5, repr.plot.height = 5)
dt_pharma_apache_count <- setDT(dt_pharma_apache)[,list(Count=.N),names(dt_pharma_apache)]
ggplot(dt_pharma_apache_count, aes(apache, pharmaid, size=Count, color=dischargestatus)) + geom_point() + theme_rotatex + scale_color_manual(values = c('black', 'red'))

dt_info[, los_larger_than_1 := dt_info$los_remain > 1]
ggplot(dt_info, aes(apache, pharmaid)) + geom_point() + theme_rotatex + facet_grid(~sex) 
ggplot(dt_info, aes(apache, fill=factor(age))) + geom_bar() + theme_rotatex
ggplot(dt_info, aes(apache, pharmaid, )) + geom_point() + theme_rotatex + facet_grid(~los_larger_than_1)

# correlation pharma - physio change
dt_change <- fread('physio_change_after_pharma.csv')
dt_change <- as.data.table(dt_change)

dt_plot = merge(dt_info, dt_change, by='sampleid')
remove(dt_info)
remove(dt_change)
remove(dt)

pharma_list <- list()
for (pharma in COL_PHARMA[1:21]){
  pharma_list <- append(pharma_list, as.integer(sub('pharma_([0-9+])', '\\1', pharma)))
}
pharma_list <- unlist(pharma_list)
  

subset <- dt_plot[pharmaid %in% pharma_list]
subset$pharmaid <- factor(subset$pharmaid, pharma_list)
subset[, los_larger_than_1 := ifelse( subset$los_remain > 1, "remained LOS > 1", "remained LOS <= 1")]

subset_corr <- subset[, .(physio_2_mean_prev, physio_17_mean_prev, physio_32_mean_prev, physio_29_mean_prev)]
subset_corr$physio_29_mean_prev[subset_corr$physio_29_mean_prev < 0.2] <- 0
subset_corr$physio_29_mean_prev[subset_corr$physio_29_mean_prev >= 0.2 & subset_corr$physio_29_mean_prev < 0.4] <- 0.2
subset_corr$physio_29_mean_prev[subset_corr$physio_29_mean_prev >= 0.4 & subset_corr$physio_29_mean_prev < 0.6] <- 0.4
subset_corr$physio_29_mean_prev[subset_corr$physio_29_mean_prev >= 0.6 & subset_corr$physio_29_mean_prev < 0.8] <- 0.6
subset_corr$physio_29_mean_prev[subset_corr$physio_29_mean_prev >= 0.8] <- 1
subset_corr$physio_29_mean_prev <- factor(subset_corr$physio_29_mean_prev)

ggpairs(na.omit(subset_corr), columnLabels = c(NAMES_PHYSIO[2], NAMES_PHYSIO[17], NAMES_PHYSIO[32], NAMES_PHYSIO[29]))


options(repr.plot.width = 3, repr.plot.height =3)

ggplot(subset, aes(pharmaid, physio_1_mean_after - physio_1_mean_prev)) + geom_boxplot() + theme_rotatex + labs(y=NAMES_PHYSIO[1]) + ylim(-0.3, 0.3) # SpO2
ggplot(subset, aes(pharmaid, physio_2_mean_after - physio_2_mean_prev)) + geom_boxplot() + theme_rotatex + labs(y=NAMES_PHYSIO[2]) + ylim(-0.3, 0.3) # RR
ggplot(subset, aes(pharmaid, physio_17_mean_after - physio_17_mean_prev)) + geom_boxplot() + theme_rotatex + labs(y=NAMES_PHYSIO[17]) + ylim(-0.3, 0.3) # HR
temperature <- ggplot(subset, aes(pharmaid, physio_18_mean_after - physio_18_mean_prev)) + geom_boxplot() + theme_rotatex + labs(y=NAMES_PHYSIO[18]) # temperature
ggplot(subset, aes(pharmaid, physio_32_mean_after - physio_32_mean_prev)) + geom_boxplot() + theme_rotatex + labs(y=NAMES_PHYSIO[32]) + ylim(-0.1, 0.1) # fluid balance
ggplot(subset, aes(pharmaid, physio_31_mean_after - physio_31_mean_prev)) + geom_boxplot() + theme_rotatex + labs(y=NAMES_PHYSIO[31]) + ylim(-0.05, 0.05) # drainage

figure <- ggarrange(spo2, rr, hr, temperature, fluid, drain,
                    ncol = 2, nrow = 3)


ggplot(subset, aes(pharmaid, physio_17_mean_after - physio_17_mean_prev, color=los_remain)) + geom_point() + theme_rotatex + labs(y=NAMES_PHYSIO[17]) + facet_grid(~los_larger_than_1) # HR

options(repr.plot.width = 8, repr.plot.height = 3)
ggplot(subset, aes(pharmaid, physio_29_mean_after - physio_29_mean_prev)) + geom_boxplot() + ylim(-0.4, 0.4) + theme_rotatex + labs(y=NAMES_PHYSIO[29]) + facet_grid(~los_larger_than_1) 
ggplot(subset, aes(pharmaid, physio_29_mean_after - physio_29_mean_prev, color=age)) + geom_point() + theme_rotatex + labs(y=NAMES_PHYSIO[29])  + facet_grid(~los_larger_than_1) 

# subset_scale_los <- copy(subset)
# subset_scale_los$los_remain <- log10(subset_scale_los$los_remain)
# ggplot(subset, aes(pharmaid, physio_29_mean_after - physio_29_mean_prev, color=los_remain)) + geom_point() + theme_rotatex + labs(y=NAMES_PHYSIO[29]) + scale_color_brewer()

ggplot(subset, aes(pharmaid, physio_26_mean_after - physio_26_mean_prev, color=age)) + geom_boxplot() + theme_rotatex + labs(y=NAMES_PHYSIO[26]) + facet_grid(~los_larger_than_1) 
ggplot(subset, aes(pharmaid, physio_26_mean_after - physio_26_mean_prev)) + geom_point() + facet_grid(~sex) + theme_rotatex + labs(y=NAMES_PHYSIO[26]) 

options(repr.plot.width = 3, repr.plot.height = 3)
dt85 <- subset[subset$pharmaid==85]
dt85_hr <- melt(dt85, id.vars = c(COL_INFO, 'physio_17_mean_prev'), 
                measure.vars = c('physio_17_mean_after_0_20', 'physio_17_mean_after_20_40', 'physio_17_mean_after_40_60'), 
                variable.name = 'HR_after', value.name = 'value')
ggplot(dt85_hr, aes(physio_17_mean_prev, value, color=HR_after)) + geom_abline() + geom_point() + labs(x='HR before medicine', y='HR after medicine') + geom_smooth(method = 'lm', fullrange = TRUE) + scale_x_continuous(expand = c(0,0), limits=c(0,1)) + scale_y_continuous(expand = c(0,0), limits=c(0,1)) + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min'))   # HR


dt85_fluid <- melt(dt85, id.vars = c(COL_INFO, 'physio_32_mean_prev'), 
                measure.vars = c('physio_32_mean_after_0_20', 'physio_32_mean_after_20_40', 'physio_32_mean_after_40_60'), 
                variable.name = 'Fluid_Balance_after', value.name = 'value')
ggplot(dt85_fluid, aes(physio_32_mean_prev, value, color=Fluid_Balance_after)) + geom_abline() + geom_point() + labs(x='Fluid balance before medicine', y='Fluid balance after medicine') + geom_smooth(method = 'lm', fullrange = TRUE) + scale_x_continuous(expand = c(0,0), limits=c(0,0.35)) + scale_y_continuous(expand = c(0,0), limits=c(0,0.35)) + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min'))   # fluid balance

dt85_rr <- melt(dt85, id.vars = c(COL_INFO, 'physio_2_mean_prev'), 
                   measure.vars = c('physio_2_mean_after_0_20', 'physio_2_mean_after_20_40', 'physio_2_mean_after_40_60'), 
                   variable.name = 'RR_after', value.name = 'value')
ggplot(dt85_rr, aes(physio_2_mean_prev, value, color=RR_after)) + geom_point() + labs(x='RR before medicine', y='RR after medicine') + geom_smooth(method = 'lm', fullrange = TRUE) + scale_x_continuous(expand = c(0,0), limits=c(0,1)) + scale_y_continuous(expand = c(0,0), limits=c(0,1)) + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min'))   # RR

dt85_richmond <- melt(dt85, id.vars = c(COL_INFO, 'physio_29_mean_prev'), 
                measure.vars = c('physio_29_mean_after_0_20', 'physio_29_mean_after_20_40', 'physio_29_mean_after_40_60'), 
                variable.name = 'Richmond_after', value.name = 'value')
ggplot(dt85_richmond, aes(physio_29_mean_prev, value, color=Richmond_after)) + geom_point() + labs(x='Richmond before medicine', y='Richmond after medicine') + geom_smooth(method = 'lm', fullrange = TRUE) + scale_x_continuous(expand = c(0,0), limits=c(0,1)) + scale_y_continuous(expand = c(0,0), limits=c(0,1)) + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min'))   # richmond agitation-sedation scale



dt15 <- subset[subset$pharmaid==15]
dt15_richmond <- melt(dt85, id.vars = c(COL_INFO, 'physio_29_mean_prev'), 
                      measure.vars = c('physio_29_mean_after_0_20', 'physio_29_mean_after_20_40', 'physio_29_mean_after_40_60'), 
                      variable.name = 'Richmond_after', value.name = 'value')
ggplot(dt15_richmond, aes(physio_29_mean_prev, value, color=Richmond_after)) + geom_point() + labs(x='Richmond before medicine', y='Richmond after medicine') + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min'))   # richmond agitation-sedation scale
ggplot(dt15_richmond, aes(physio_29_mean_prev, value, color=Richmond_after)) + geom_abline() + geom_jitter(width = 0.02, height = 0.02) + labs(x='Richmond before medicine', y='Richmond after medicine') + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min')) + theme_bw() # richmond agitation-sedation scale
ggplot(dt15_richmond, aes(physio_29_mean_prev, value, color=Richmond_after)) + geom_jitter(width = 0.02, height = 0.02) + labs(x='Richmond before medicine', y='Richmond after medicine') + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min')) + facet_grid(~age)   # richmond agitation-sedation scale



dt1000763 <- subset[subset$pharmaid==1000763]
dt1000763_fluid <- melt(dt1000763, id.vars = c(COL_INFO, 'physio_32_mean_prev'), 
                   measure.vars = c('physio_32_mean_after_0_20', 'physio_32_mean_after_20_40', 'physio_32_mean_after_40_60'), 
                   variable.name = 'Fluid_Balance_after', value.name = 'value')
ggplot(dt85_fluid, aes(physio_32_mean_prev, value, color=Fluid_Balance_after)) + geom_point() + labs(x='Fluid balance before medicine', y='Fluid balance after medicine') + geom_smooth(method = 'lm', fullrange = TRUE) + scale_x_continuous(expand = c(0,0), limits=c(0,0.35)) + scale_y_continuous(expand = c(0,0), limits=c(0,0.35)) + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min'))   # fluid balance
dt1000763_hr <- melt(dt1000763, id.vars = c(COL_INFO, 'physio_17_mean_prev'), 
                measure.vars = c('physio_17_mean_after_0_20', 'physio_17_mean_after_20_40', 'physio_17_mean_after_40_60'), 
                variable.name = 'HR_after', value.name = 'value')
ggplot(dt1000763_hr, aes(physio_17_mean_prev, value, color=HR_after)) + geom_point() + labs(x='HR before medicine', y='HR after medicine') + geom_smooth(method = 'lm', fullrange = TRUE) + scale_x_continuous(expand = c(0,0), limits=c(0,1)) + scale_y_continuous(expand = c(0,0), limits=c(0,1)) + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min')) + geom_abline()    # HR
dt1000763_rr <- melt(dt1000763, id.vars = c(COL_INFO, 'physio_2_mean_prev'), 
                measure.vars = c('physio_2_mean_after_0_20', 'physio_2_mean_after_20_40', 'physio_2_mean_after_40_60'), 
                variable.name = 'RR_after', value.name = 'value')
ggplot(dt1000763_rr, aes(physio_2_mean_prev, value, color=RR_after)) + geom_point() + labs(x='RR before medicine', y='RR after medicine') + geom_smooth(method = 'lm', fullrange = TRUE) + scale_x_continuous(expand = c(0,0), limits=c(0,1)) + scale_y_continuous(expand = c(0,0), limits=c(0,1)) + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min'))   # RR
dt1000763_richmond <- melt(dt1000763, id.vars = c(COL_INFO, 'physio_29_mean_prev'), 
                      measure.vars = c('physio_29_mean_after_0_20', 'physio_29_mean_after_20_40', 'physio_29_mean_after_40_60'), 
                      variable.name = 'Richmond_after', value.name = 'value')
ggplot(dt1000763_richmond, aes(physio_29_mean_prev, value, color=Richmond_after)) + geom_jitter(width = 0.02, height = 0.02) + labs(x='Richmond before medicine', y='Richmond after medicine') + theme_bw() + scale_color_discrete(labels=c('0-20 min', '20-40 min', '40-60 min'))   # richmond agitation-sedation scale



ggplot(dt_change, aes(physio_1_mean_prev, physio_1_mean_after)) + geom_point() + theme_bw()
ggplot(dt_change, aes(physio_1_mean_prev, physio_1_mean_after_0_20)) + geom_point() + theme_bw()
ggplot(dt_change, aes(physio_1_mean_prev, physio_1_mean_after_20_40)) + geom_point() + theme_bw()

ggplot((dt_change, aes(physio_1_mean_prev)))

ggplot(dt_change, aes(physio_1_mean_prev, physio_1_mean_after)) + geom_point() + theme_bw()
ggplot(dt_change, aes(physio_2_mean_prev, physio_2_mean_after)) + geom_point() + theme_bw()
