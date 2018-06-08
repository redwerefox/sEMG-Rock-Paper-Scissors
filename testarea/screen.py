from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

# Declare both screens
class MenuScreen(Screen):
	pass

class SettingsScreen(Screen):
	pass



class TestApp(App):

	def build(self):
		buildKV = Builder.load_file("screen.kv")	
	
		# Create the screen manager
		sm = ScreenManager()
		sm.add_widget(MenuScreen(name='menu'))
		sm.add_widget(SettingsScreen(name='settings'))


		return sm

if __name__ == '__main__':
	TestApp().run()