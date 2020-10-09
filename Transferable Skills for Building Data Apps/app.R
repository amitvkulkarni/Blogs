
#install.packages("shiny")
library(shiny)

#load the dataset
df = read.csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')


# Define UI for application 
ui <- fluidPage(
    
        # Show a plot of the generated distribution
        mainPanel(
            
           plotOutput("distPlot"),
           sliderInput("yearSelected","Years:",
                       min = min(unique(df['year'])),
                       max = max(unique(df['year'])),
                       value = 1972, step = 5, width = '1200px')
        )
)

# Define server logic
server <- function(input, output) {
    
    output$distPlot <- renderPlot({
        
        # filter the data for selected year from the slider
        df_yearfiltered <- df %>% filter(df['year'] == input$yearSelected)
        #prep the data for plot
        df_plot <- df_yearfiltered %>%
                        group_by(df_yearfiltered['country']) %>%
                        summarise('lifeExp' = mean(lifeExp)) %>%
                        arrange("lifeExp") %>%
                        top_n(20) # displaying only top 20 countries

        # # plot the bar chart and set aesthetics
        ggplot(data = df_plot,aes(x = country, y = lifeExp)) +
                        geom_bar(stat="identity", fill="steelblue") +
                        labs(title = "Bar plot: Country vs LifeExp") +
                        theme(plot.title = element_text(size=22))

        
    })

}

# Run the application 
shinyApp(ui = ui, server = server)

df %>% 
    group_by(df['country']) %>% 
    summarise('life' = count(df['lifeExp']))
