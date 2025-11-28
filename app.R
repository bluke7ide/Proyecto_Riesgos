library(shiny)
library(tidyverse)
library(isotree)
library(recipes)
library(torch)
library(readr)
library(bslib)


bank <- read_csv("data/bank_transactions_data_2.csv")

features <- bank %>%
  select(TransactionAmount, AccountBalance,
         TransactionDuration, LoginAttempts, CustomerAge)

rec <- recipe(~ ., data = features) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric())

prep_rec <- prep(rec, training = features)
features_scaled <- bake(prep_rec, new_data = features)

mean_amount <- mean(bank$TransactionAmount)
sd_amount   <- sd(bank$TransactionAmount)
z_threshold <- mean_amount + 2 * sd_amount
bank$Z_flag <- bank$TransactionAmount > z_threshold

iso_model <- isolation.forest(as.matrix(features_scaled), ntrees = 100)
iso_scores <- predict(iso_model, as.matrix(features_scaled), type = "score")
iso_threshold <- quantile(iso_scores, 0.95)
bank$iso_flag <- iso_scores > iso_threshold
bank$iso_score <- iso_scores

x <- torch_tensor(as.matrix(features_scaled), dtype = torch_float())
input_dim <- ncol(features_scaled)
encoder_dim <- round(input_dim / 2)

autoencoder <- nn_module(
  initialize = function() {
    self$encoder <- nn_linear(input_dim, encoder_dim)
    self$decoder <- nn_linear(encoder_dim, input_dim)
  },
  forward = function(x) {
    x %>% self$encoder() %>% nnf_relu() %>% self$decoder()
  }
)

model <- autoencoder()
optimizer <- optim_adam(model$parameters, lr = 0.001)
loss_fn <- nn_mse_loss()

for (epoch in 1:40) {
  optimizer$zero_grad()
  loss <- loss_fn(model(x), x)
  loss$backward()
  optimizer$step()
}

recon <- as.array(model(x))
mse_train <- rowMeans((as.matrix(features_scaled) - recon)^2)
ae_threshold <- quantile(mse_train, 0.95)
bank$ae_flag <- mse_train > ae_threshold
bank$ae_mse  <- mse_train


theme <- bs_theme(
  bootswatch = "flatly",
  primary = "#0b5ed7",
  base_font = font_google("Inter")
)


ui <- fluidPage(
  theme = theme,
  
  tags$head(
    tags$style(HTML("
      body {
        background: linear-gradient(135deg, #eef3f9, #dbe7f5);
      }
      .sidebar {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      }
      .main-panel {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      }
      h3, h4 {
        font-weight: 600;
      }
    "))
  ),
  
  titlePanel("Sistema de Detección de Anomalías en Transacciones"),
  
  sidebarLayout(
    sidebarPanel(
      class = "sidebar",
      
      numericInput("TransactionAmount", "Monto de Transacción", 500),
      numericInput("AccountBalance", "Balance de la Cuenta", 2000),
      numericInput("TransactionDuration", "Duración (segundos)", 60),
      numericInput("LoginAttempts", "Intentos de Login", 1),
      numericInput("CustomerAge", "Edad del Cliente", 30)
    ),
    
    mainPanel(
      class = "main-panel",
      
      tabsetPanel(
        tabPanel("Z-Score",
                 strong(textOutput("z_text")),
                 textOutput("z_ref"),
                 plotOutput("z_plot", height = 360)
        ),
        
        tabPanel("Isolation Forest",
                 strong(textOutput("iso_text")),
                 textOutput("iso_score_text"),
                 plotOutput("iso_plot", height = 360)
        ),
        
        tabPanel("Autoencoder",
                 strong(textOutput("ae_text")),
                 textOutput("ae_score_text"),
                 plotOutput("ae_plot", height = 360)
        )
      )
    )
  )
)

server <- function(input, output) {
  
  new_point <- reactive({
    tibble(
      TransactionAmount   = input$TransactionAmount,
      AccountBalance      = input$AccountBalance,
      TransactionDuration = input$TransactionDuration,
      LoginAttempts       = input$LoginAttempts,
      CustomerAge         = input$CustomerAge
    )
  })
  
  new_scaled <- reactive({
    bake(prep_rec, new_point())
  })
  
  # FLAGS
  z_flag <- reactive(input$TransactionAmount > z_threshold)
  
  iso_score_new <- reactive(
    predict(iso_model, as.matrix(new_scaled()), type = "score")
  )
  
  iso_flag <- reactive(iso_score_new() > iso_threshold)
  
  ae_mse_new <- reactive({
    x_new <- torch_tensor(as.matrix(new_scaled()), dtype = torch_float())
    recon <- as.array(model(x_new))
    mean((as.matrix(new_scaled()) - recon)^2)
  })
  
  ae_flag <- reactive(ae_mse_new() > ae_threshold)
  
  # TEXT
  output$z_text <- renderText(
    if (z_flag()) "Clasificación: Transacción Atípica"
    else "Clasificación: Transacción Consistente"
  )
  
  output$z_ref <- renderText(
    paste("Umbral de referencia:", round(z_threshold, 2))
  )
  
  output$iso_text <- renderText(
    if (iso_flag()) "Clasificación: Transacción Atípica"
    else "Clasificación: Transacción Consistente"
  )
  
  output$iso_score_text <- renderText(
    paste("Score:", round(iso_score_new(), 4),
          "| Umbral:", round(iso_threshold, 4))
  )
  
  output$ae_text <- renderText(
    if (ae_flag()) "Clasificación: Transacción Atípica"
    else "Clasificación: Transacción Consistente"
  )
  
  output$ae_score_text <- renderText(
    paste("MSE:", round(ae_mse_new(), 4),
          "| Umbral:", round(ae_threshold, 4))
  )
  
  # PLOTS
  base_plot <- function(flag, color) {
    ggplot(bank, aes(TransactionAmount, AccountBalance)) +
      geom_point(aes(color = factor(flag)), alpha = 0.2) +
      annotate(
        "point",
        x = input$TransactionAmount,
        y = input$AccountBalance,
        color = color,
        size = 4
      ) +
      scale_color_manual(
        values = c("FALSE" = "#d7301f", "TRUE" = "#2c7fb8"),
        labels = c("Consistente", "Atípica")
      ) +
      theme_minimal()
  }
  
  
  output$z_plot <- renderPlot(
    base_plot(
      bank$Z_flag,
      ifelse(z_flag(), "#2c7fb8", "#d7301f")
    )
  )
  
  output$iso_plot <- renderPlot(
    base_plot(
      bank$iso_flag,
      ifelse(iso_flag(), "#2c7fb8", "#d7301f")
    )
  )
  
  output$ae_plot <- renderPlot(
    base_plot(
      bank$ae_flag,
      ifelse(ae_flag(), "#2c7fb8", "#d7301f")
    )
  )
  
}

shinyApp(ui, server)
