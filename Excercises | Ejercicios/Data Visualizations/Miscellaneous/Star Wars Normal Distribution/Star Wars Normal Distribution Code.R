library(showtext)
library(sysfonts)

font_add(family = "StarJedi", regular = "C:/Users/phuet/Downloads/star_jedi/starjedi/Starjedi.ttf")

showtext_auto()

library(ggplot2)

episode_names <- c(
  "Episode I: The Phantom Menace", 
  "Episode II: Attack of the Clones", 
  "Episode III: Revenge of the Sith", 
  "Episode IV: A New Hope", 
  "Episode V: The Empire Strikes Back", 
  "Episode VI: Return of the Jedi", 
  "Episode VII: The Force Awakens", 
  "Episode VIII: The Last Jedi", 
  "Episode IX: The Rise of Skywalker"
)

episodes <- 1:9
mean_ep <- 5
sd_ep <- 1.5
cinematic_quality <- dnorm(episodes, mean = mean_ep, sd = sd_ep)

cinematic_quality_scaled <- (cinematic_quality / max(cinematic_quality)) * 10

df <- data.frame(
  EpisodeName = factor(episode_names, levels = episode_names),
  CinematicQuality = cinematic_quality_scaled
)

p <- ggplot(df, aes(x = EpisodeName, y = CinematicQuality)) +
  geom_col(fill = "cyan", width = 0.5) +  
  labs(
    title = "Star Wars Episodes (by quality)",
    x = NULL,
    y = NULL
  ) +
  scale_y_continuous(
    breaks = c(0.5, 1.75, 4.4, 7, 10),
    labels = c("We don't talk about it", "Subpar", "Good", "Great", "Classic"),
    limits = c(0, 10)
  )+
  theme_minimal(base_family = "") +
  theme(
    plot.background = element_rect(fill = "black", color = NA),
    panel.background = element_rect(fill = "black", color = NA),
    panel.grid = element_blank(),
    axis.text = element_text(color = "white"),
    axis.title.x = element_text(color = "white"),
    plot.title = element_text(
      family = "StarJedi",
      size = 16,
      color = "yellow",
      face = "bold",
      hjust = 0.5
    ),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p <- p + 
  annotate(
    "label",
    x = 9,                
    y = 10,               
    label = "Another Normal Distribution from nature...",
    hjust = 1,
    vjust = 1,
    fill = "gray20",
    color = "yellow",
    label.size = 0.2,
    size = 3.5
  )

ggsave("Star_Wars_Cinematic_Quality_Custom_Y_Axis.pdf", plot = p, width = 10, height = 6)
